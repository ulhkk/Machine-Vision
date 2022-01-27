import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='')

    #  experiment settings
    parser.add_argument('--root_dir', 
                        default='/home/guanzhi/data/direction_learning/dataset', 
                        type=str,
                        help='dataset directory')

    parser.add_argument('--batch_size', 
                        default=8, 
                        type=int,
                        help='training batch size')

    parser.add_argument('--lr', 
                        default=1e-3, 
                        type=float,
                        help='learning rate')
                
    parser.add_argument('--epochs', 
                        default=200, 
                        type=int,
                        help='number of training epochs')

    parser.add_argument('--save_model_path', 
                        default='/home/guanzhi/data/direction_learning/', 
                        type=str,
                        help='checkpoints dir')

    parser.add_argument('--weight_decay', 
                        default=1e-5, 
                        type=float,
                        help='weight_decay')
    
    parser.add_argument('--exp_dir', 
                        default='/home/guanzhi/data/direction_learning/exp', 
                        type=str,
                        help='save dir')

    args = parser.parse_args()
    
    assert args.root_dir is not None
    
    print(' '.join(sys.argv))
    print(args)

    return args
