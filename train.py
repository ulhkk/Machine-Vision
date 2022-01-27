import numpy as np
import math
import random
import os
import torch
import tqdm
import matplotlib.pyplot as plt
from path import Path
from source import model as custModel
from source import data_loader
from source import utils
from source.args import parse_args
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

random.seed = 42

class sin_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return torch.pow(torch.sin((x - y) / 2), 2)

def print_info(curr_epoch, train_loss_per_epoch, train_accuracy_per_epoch, validation_loss_per_epoch, val_accuracy_per_epoch):
    print(
            "Epoch number {}, Current train loss {}, Current_train_accuracy {}, Current validation loss {}, Current_validation_accuracy {}, ".format(
            curr_epoch, train_loss_per_epoch, train_accuracy_per_epoch, validation_loss_per_epoch, val_accuracy_per_epoch)
            )

def load_data(path, data_mode, batch_size, data_augumentation):
    cloud_dataset = data_loader.PointCloudData(root_dir=path, mode=data_mode, data_argumentation= data_augumentation)
    cloud_loader = torch.utils.data.DataLoader(cloud_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
    return cloud_loader

def pointnetloss(outputs, yaw, m3x3, m64x64, alpha=0.0001):
    # criterion = torch.nn.HuberLoss()
    criterion = abs(math.sin((outputs - yaw) / 2))
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
    return criterion + alpha * (torch.norm(diff3x3) +
                                torch.norm(diff64x64)) / float(bs)
    # 第二项是用了旋转矩阵性质(转置乘以本身为单位阵)，使stn结果总是更接近一个旋转矩阵
    # norm 平法差的和

def run_epoch(data_loader, model, mode, learning_rate, weight_decay, use_tqdm = False):
    assert mode in ['Train','Validation']
    if mode == 'Train':
        model.train()
    else:
        model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay= weight_decay)
    criterion = torch.nn.CrossEntropyLoss() #custModel.get_loss()

    loss_list = []
    accuracy_list = []
    total = 0
    co = 0

    for i, data in enumerate(tqdm.tqdm(data_loader, 0)):
        cloud, label = data

       # cloud = cloud.permute(0,2,1)
        if(cloud.size() == torch.Size([0,0,0]) or cloud.size() == torch.Size([0,0]) or list(cloud.size())[2] <= 1 ): continue
        if torch.cuda.is_available():
            cloud, label = cloud.cuda(), label.cuda()
        
        if mode == 'Train':
            optimizer.zero_grad()
            predict = model.forward(cloud)
            #predict = predict.float()
            _, pre = torch.max(predict.data, dim = 1)
            total += label.size(0)
            co+=(pre == label).sum().item()
            loss_ = criterion(predict, label)
            loss_.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                predict = model.forward(cloud)
                loss_ = criterion(predict, label)
                _, pre = torch.max(predict.data, dim = 1)
                total += label.size(0)
                co+=(pre == label).sum().item()
        pred_choice = predict.data.max(1)[1]
        correct = pred_choice.eq(label.data).cpu().sum()
        loss_list.append(loss_.item())
        accuracy_list.append(correct // args.batch_size) 
        #import pdb; pdb.set_trace()
    # print(co / total)
    # print('\n')
    loss_return = np.mean(np.asarray(loss_list))
    accuracy_return = np.mean(np.asarray(accuracy_list))
    return loss_return, accuracy_return

def train(args):
    path = Path(args.root_dir)

    train_loader = load_data(path, data_mode='Train',batch_size=args.batch_size, data_augumentation=True)
    print("--------training data loaded-------")
    val_loader = load_data(path, data_mode='Validation', batch_size=args.batch_size, data_augumentation=False)
    print("--------validation data loaded-------")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pointnet = custModel.PointNet()
    pointnet.to(device)
    print(device)
    # optimizer = torch.optim.Adam(pointnet.parameters(), lr=args.lr)

    print("============Start to train===============")
    
    train_loss_history = []
    validation_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []
    counter = []
    best_validation_loss = float('inf')

    print('Train dataset size: ', len(train_loader))
    print('Valid dataset size: ', len(val_loader))

    try:
        os.mkdir(args.save_model_path)
    except OSError as error:
        print(error)

    print('Start training')
    for epoch in range(args.epochs):
        counter.append(epoch)
        train_loss_per_epoch , train_acc = run_epoch(data_loader = train_loader, model = pointnet, mode='Train', learning_rate=args.lr, weight_decay=args.weight_decay)
        validation_loss_per_epoch, validation_acc = run_epoch(data_loader = val_loader, model = pointnet, mode='Validation', learning_rate=args.lr, weight_decay=args.weight_decay)
        
        train_loss_history.append(train_loss_per_epoch)
        validation_loss_history.append(validation_loss_per_epoch)
        train_accuracy_history.append(train_acc)
        val_accuracy_history.append(validation_acc)
        
        if(epoch % 10 == 0):
            print_info(epoch, train_loss_per_epoch, train_acc, validation_loss_per_epoch, validation_acc)
        
        if validation_loss_per_epoch < best_validation_loss:
            save_model_path = '/home/guanzhi/data/direction_learning/best_model.pth'
            torch.save(pointnet.state_dict(), save_model_path)

    plt.subplots()
    plt.plot(counter[10:], train_loss_history[10:], label = 'train loss')
    plt.plot(counter[10:], validation_loss_history[10:], label = 'validation loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss diagramm')
    fig_file_name = 'current_loss.png'
    save_path = os.path.join(args.exp_dir + '/train/', fig_file_name)
    plt.savefig(args.exp_dir + '/train/' + fig_file_name)  

    plt.subplots() 
    plt.plot(counter,train_accuracy_history, label = 'train accuracy')
    plt.plot(counter,val_accuracy_history, label = 'validation accuracy')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy diagramm')
    fig_file_name = 'current_accuracy.png'
    save_path = os.path.join(args.exp_dir + '/train/', fig_file_name)
    plt.savefig(args.exp_dir + '/train/' + fig_file_name)  
    
    plt.close('all')

if __name__ == '__main__':
    args = parse_args()
    train(args)
    # pointnet = model.PointNet()
    # input = torch.rand([2,4,1])
    # print(input.size())
    # print(input[0].size())
    # print(input[0][0].size() == torch.Size([1]))
    # pointnet.forward(input)