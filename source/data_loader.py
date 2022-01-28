from cgi import test
from cmath import pi
import os
import torch
import yaml
import numpy as np
import tqdm
import utils
import random
import math
from path import Path
from torchvision import transforms
# 数据的格式的变换

class PointCloudData(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode="Train", data_argumentation=True, transform=None, num_points=30):
        assert mode in ['Train', 'Validation', 'Test']
        self.root_dir_ = root_dir
        self.transform_ = transform
        self.num_points_ = num_points
        self.data_mode_ = mode
        self.argumentation_ = data_argumentation
        self.points_ = []
        self.labels_ = []

        #read
        folder_data = self.root_dir_ + '/' + self.data_mode_ + '/data'
        folder_label = self.root_dir_ + '/' + self.data_mode_ + '/label'
        files_data = os.listdir(folder_data)
        files_label = os.listdir(folder_label)
        files_data.sort()
        files_label.sort()

        for i in range(len(files_data)):
            file_data = files_data[i]
            file_label = files_label[i]

            data_path = os.path.join(folder_data, file_data)
            label_path = os.path.join(folder_label, file_label)
            points = np.fromfile(data_path ,dtype=np.float32).reshape(-1,4)
            points = points[:, :3]
            num = points[:,0].size
            if(num == 0):
                continue
            #resize to 100
            self.points_.append(points)

            #label
            with open(label_path, 'r') as f:
                content_as_dict = yaml.safe_load(f.read())
                yaw = content_as_dict['Yaw']
                self.labels_.append(np.float32(yaw))
        print('checkpoint here')
        # self.points_ = self.points_[:1000]
        # self.labels_ = self.labels_[:1000]

    def __getitem__(self, idx):
        cloud = self.points_[idx]# Nx4
        yaw = self.labels_[idx]
        num = cloud[:,0].size
        # data augmentation
        if self.argumentation_:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            cloud[:,[0,2]] = cloud[:,[0,2]].dot(rotation_matrix) # random rotation
            cloud += np.random.normal(0, 0.02, size=cloud.shape) # random jitter
            # yaw += theta
            # if yaw > np.pi : 
            #     yaw = yaw - 2 * np.pi
            # elif yaw < - np.pi :
            #     yaw = yaw + 2 * np.pi
    
        yaw = int(np.float32(yaw * 180 / math.pi / 10)) + 18

        if num >= self.num_points_:
            selected_points_idx = np.random.choice(np.arange(0, num), self.num_points_, replace=False)
        else:
            selected_points_idx = np.random.choice(np.arange(0, num), self.num_points_, replace=True)
        cloud = cloud[selected_points_idx]   
         #cloud = np.pad(cloud, ((0,self.num_points_ - num),(0,0)),'constant', constant_values = 0)

        return cloud.T, yaw

    def __len__(self):
        return len(self.labels_)

    def __preproc__(self, file):
        verts, faces = utils.read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

if __name__ == '__main__':

    print(int(np.float32(-3.04 * 180 / math.pi)))
    # test_data = [1,2,3,4,5,6,7]
    # delete_list = [1,3]
    # for item in delete_list:
    #     del test_data[item]
    # print(test_data)
    # print(test_data.size)
    # print(test_data[:,0].size)
    # test_data.shape = 3,4
    # print(test_data)
    # print(test_data.dtype)
    # test_data.tofile('/home/guanzhi/liangdao/pointnet/Train/points/gener.bin')
    # test_data2 = np.arange(5,17)
    # test_data2.shape = 3,4
    # print(test_data2)
    # test_data2.tofile('/home/guanzhi/liangdao/pointnet/Train/points/gener2.bin')
    # test_dataset = PointCloudData(root_dir='/home/guanzhi/liangdao/pointnet', data_argumentation=True)
    # test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = True, num_workers = 8)

    # for i, data in enumerate(tqdm.tqdm(test_data_loader, 0)):
    #     cloud, yaw = data
    #     print(cloud.size())
    pass