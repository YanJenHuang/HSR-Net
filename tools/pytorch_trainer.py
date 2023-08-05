import abc
import numpy as np
import matplotlib.pyplot as plt

import torch

class TrainerBase(abc.ABC):
    def __init__(self, model, scheduler, criterion, optimizer, device, verbose, record_period, name=None):
        self.model = model
        self.scheduler = scheduler
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.verbose = verbose
        self.record_period = record_period
        self.name = name
    
        self.recorded_acc_dict = {'num_epoch': [],
                             'train_acc': [],
                             'valid_acc': []}
        self.recorded_loss_dict = {'num_parameters':None,
                              'num_epoch': [],
                              'train_loss': [],
                              'valid_loss': []}

    @abc.abstractmethod
    def start_trainning_process(self, num_epoch, train_loader, valid_loader):
        raise NotImplementedError
    
    @abc.abstractmethod
    def run_epoch(self, epoch, data_loader, criterion, optimizer):
        raise NotImplementedError
    
    @abc.abstractmethod
    def evaluation(self, data_loader):
        raise NotImplementedError
    
    def calculate_accuracy(self, output, target):
        """
        Top-1 classification accuracy.
        """
        with torch.no_grad():
            return torch.abs(output-target).float().mean()
    
    def record_accuracy(self, epoch, train_acc, valid_acc):
        self.recorded_acc_dict['num_epoch'].append(epoch)
        self.recorded_acc_dict['train_acc'].append(train_acc)
        self.recorded_acc_dict['valid_acc'].append(valid_acc)
        
    def record_loss(self, epoch, train_loss, valid_loss):
        self.recorded_loss_dict['num_epoch'].append(epoch)
        self.recorded_loss_dict['train_loss'].append(train_loss)
        self.recorded_loss_dict['valid_loss'].append(valid_loss)
        
    def plot_accuracy(self, title=None, mode=0):
        plt.plot(self.recorded_acc_dict['num_epoch'], self.recorded_acc_dict['train_acc'], color='tab:blue', label='train_accuracy')
        plt.plot(self.recorded_acc_dict['num_epoch'], self.recorded_acc_dict['valid_acc'], color='tab:orange', label='valid_accuracy')
        plt.xlabel('Number of epoch')
        plt.ylabel('Accuracy (percentage)')
        if title is not None:
            plt.title(title)
        elif self.name is not None:
            plt.title(self.name)
        else:
            pass
        plt.legend()
        plt.show()
    
    def plot_loss(self, title=None, mode=0):
        plt.plot(self.recorded_loss_dict['num_epoch'], self.recorded_loss_dict['train_loss'], color='tab:blue', label='train_loss')
        plt.plot(self.recorded_loss_dict['num_epoch'], self.recorded_loss_dict['valid_loss'], color='tab:orange', label='valid_loss')
        plt.xlabel('Number of epoch')
        plt.ylabel('Accuracy (percentage)')
        if title is not None:
            plt.title(title)
        elif self.name is not None:
            plt.title(self.name)
        else:
            pass
        plt.legend()
        plt.show()
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        print('Total parameters: {:,}\nTrainable parameters: {:,}\nNon-trainable parameters: {:,}'.format(
                total_params, trainable_params, non_trainable_params))
        return total_params, trainable_params, non_trainable_params

    def save_records(self, name=None):
        if name is not None:
            pass
        elif self.name is not None:
            name = self.name
        else:
            raise ValueError('"name" should be given.')
        
        if len(self.recorded_acc_dict['num_epoch']) is not 0:
            np.save('{}_recorded_acc'.format(name), self.recorded_acc_dict)
        
        self.recorded_loss_dict['num_parameters'] = self.count_parameters()
        np.save('{}_recorded_loss'.format(name), self.recorded_loss_dict)
    
    def load_model_parameters(self, path):
        self.model.load_state_dict(torch.load(path))
        
    def save_model_parameters(self, name=None):
        if name is not None:
            pass
        elif self.name is not None:
            name = self.name
        else:
            raise ValueError('"name" should be given.')
            
        filename = '{}.pth'.format(name)
        torch.save(self.model.state_dict(), filename)

class Trainer(TrainerBase):
    def __init__(self, model, scheduler, criterion, optimizer, device='cpu', verbose=0, record_period=1):
        super(Trainer, self).__init__(model, scheduler, criterion, optimizer, device, verbose, record_period)
    
    def start_trainning_process(self, num_epoch, train_loader, valid_loader):
        raise NotImplementedError
        
    def run_epoch(self, epoch, data_loader, criterion, optimizer):
        raise NotImplementedError
    
    def evaluation(self, data_loader):
        raise NotImplementedError
