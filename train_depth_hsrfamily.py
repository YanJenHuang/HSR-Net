import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nets.depth_net import HSRNetDepth31, HSRNetDepth32
from datasets.face_datasets import load_dataset
from tools.my_trainer import HSRTrainer

def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db", type=str, required=True,
                        help="database name")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=160,
                        help="number of epochs")
    parser.add_argument("--net", type=str, default='hsr_31depth',
                        help="hsr_31depth, or hsr_32depth")
    parser.add_argument("--nb_kernels", type=int, default=30,
                        help="number of kernels for hsrnet")
    parser.add_argument("--hsr_compress", type=int, default=0,
                        help="hsrnet using separable conv2d")
    parser.add_argument("--out_channels", type=int, default=10,
                        help="number of kernels for subnet")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    db_name = args.db
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    net = args.net
    nb_kernels = args.nb_kernels
    hsr_compress = args.hsr_compress
    out_channels = args.out_channels
    
    model_checkpoint_path = 'records/model_depth_checkpoints/'
    model_history_record_path = 'records/model_depth_history_records/'

    if not os.path.exists(model_checkpoint_path):
        os.makedirs(model_checkpoint_path)
    if not os.path.exists(model_history_record_path):
        os.makedirs(model_history_record_path)
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    trainset, validset = load_dataset(root='datasets', db_name=db_name) # use default transformation
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0)

    
    device = torch.device('cuda:0')
    criterion = nn.L1Loss()
    
    if net == 'hsr_31depth':
        model = HSRNetDepth31(kernel_size=nb_kernels, compress=hsr_compress, out_channels=out_channels, bin_weight_grad=True)
        optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        my_trainer = HSRTrainer(model=model,scheduler=scheduler,criterion=criterion,optimizer=optimizer,device=device)
    elif net == 'hsr_32depth':
        model = HSRNetDepth32(kernel_size=nb_kernels, compress=hsr_compress, out_channels=out_channels, bin_weight_grad=True)
        optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        my_trainer = HSRTrainer(model=model,scheduler=scheduler,criterion=criterion,optimizer=optimizer,device=device)
        

    model.to(device)

    my_trainer.count_parameters()
    
    my_trainer.start_trainning_process(nb_epochs, trainloader, validloader)
    
    model_name = model.get_model_name()
    save_file_name = model_name +'_'+ db_name
    my_trainer.save_records(model_history_record_path+save_file_name)
    my_trainer.save_model_parameters(model_checkpoint_path+save_file_name)

if __name__ == '__main__':
    main()
