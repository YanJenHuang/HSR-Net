import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets.face_datasets import load_dataset
from nets.net import HSRNet, HSRNetContext
from tools.my_trainer import HSRTrainer, HSRContextTrainer

def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age and gender estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db", type=str, required=True,
                        help="database name")
    parser.add_argument("--pipeline", type=int, required=0,
                        help="training imdb->wiki->morph2 pipeline")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=160,
                        help="number of epochs")
    parser.add_argument("--net", type=str, default='hsr',
                        help="hsr, or hsr_context")
    parser.add_argument("--nb_kernels", type=int, default=30,
                        help="number of kernels for hsrnet")
    parser.add_argument("--hsr_compress", type=str, default=None,
                        help="hsrnet (None), sephsrnet (sep), and bsephsrnet (bsep)")
    parser.add_argument("--out_channels", type=int, default=10,
                        help="number of kernels for subnet")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    db_name = args.db
    pipeline = args.pipeline
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    net = args.net
    nb_kernels = args.nb_kernels
    hsr_compress = args.hsr_compress
    out_channels = args.out_channels
    use_checkpoint = True
    
    if pipeline == 0:
        model_checkpoint_path = 'records/model_checkpoints/'
        model_history_record_path = 'records/model_history_records/'
    else:
        model_checkpoint_path = 'records/model_pipeline_checkpoints/'
        model_history_record_path = 'records/model_pipeline_history_records/'

    if not os.path.exists(model_checkpoint_path):
        os.makedirs(model_checkpoint_path)
    if not os.path.exists(model_history_record_path):
        os.makedirs(model_history_record_path)
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    trainset, validset = load_dataset(root='datasets', db_name=db_name) # use default transformation
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(net)

    device = torch.device('cuda:0')
    criterion = nn.L1Loss()
    
    if net == 'hsr':
        model = HSRNet(kernel_size=nb_kernels, compress=hsr_compress, out_channels=out_channels, bin_weight_grad=True)
        optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        my_trainer = HSRTrainer(model=model,scheduler=scheduler,criterion=criterion,optimizer=optimizer,device=device)
        
        if pipeline == 1:
            # load pretrained model
            if db_name == 'wiki':
                model_name = model.get_model_name()
                weights_file = model_name+'_'+'imdb.pth'
                my_trainer.load_model_parameters(model_checkpoint_path+weights_file)
                print('%%% Load pretrained model... (imdb.pth)')
            elif db_name == 'morph2':
                model_name = model.get_model_name()
                weights_file = model_name+'_'+'wiki.pth'
                my_trainer.load_model_parameters(model_checkpoint_path+weights_file)
                print('%%% Load pretrained model... (wiki.pth)')
    elif net == 'hsr_context':
        if use_checkpoint == True:
            # using pipeline checkpoint
            if pipeline != 1:
                raise ValueError('Should use pipeline checkpoint. "--pipeline 1"')
            
            # load pretrained model which was trained in single resolution.
            if hsr_compress == None:
                checkpoint_path = model_checkpoint_path+'hsr'+str(nb_kernels)+'-'+str(out_channels)+'_wiki'+'.pth'
            elif hsr_compress == 'sep':
                checkpoint_path = model_checkpoint_path+'sephsr'+str(nb_kernels)+'-'+str(out_channels)+'_wiki'+'.pth'
            elif hsr_compress == 'bsep':
                checkpoint_path = model_checkpoint_path+'bsephsr'+str(nb_kernels)+'-'+str(out_channels)+'_wiki'+'.pth'
            
            print('@@@ Load checkpoints from {}'.format(checkpoint_path))
            model = HSRNetContext(kernel_size=nb_kernels, compress=hsr_compress, out_channels=out_channels, bin_weight_grad=True, checkpoint_path=checkpoint_path)
        else:
            print('Training from scratch.')
            model = HSRNetContext(kernel_size=nb_kernels, compress=hsr_compress, out_channels=out_channels, bin_weight_grad=True, checkpoint_path=None)
            
        optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, threshold=0.0005, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        my_trainer = HSRContextTrainer(model=model,scheduler=scheduler,criterion=criterion,optimizer=optimizer,device=device)

    print(model)
    model.to(device)

    my_trainer.count_parameters()
    
    my_trainer.start_trainning_process(nb_epochs, trainloader, validloader)
    
    model_name = model.get_model_name()
    save_file_name = model_name +'_'+ db_name
    my_trainer.save_records(model_history_record_path+save_file_name)
    my_trainer.save_model_parameters(model_checkpoint_path+save_file_name)

if __name__ == '__main__':
    main()
