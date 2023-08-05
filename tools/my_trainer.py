import torch
import torch.nn.functional as F
from tools.pytorch_trainer import Trainer
from tools.utils import logger, timer

class HSRContextTrainer(Trainer):
    def __init__(self, model, scheduler, criterion, optimizer, device='cpu', verbose=0, record_period=1):
        super(HSRContextTrainer, self).__init__(model, scheduler, criterion, optimizer, device, verbose, record_period)
    
    @logger
    def start_trainning_process(self, num_epoch, train_loader, valid_loader):
        for epoch in range(num_epoch):
            train_loss = self.run_epoch(epoch, train_loader)

            train_loss = self.evaluation(train_loader)
            valid_loss = self.evaluation(valid_loader)
                
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics=valid_loss)
            else:
                self.scheduler.step()
                
            print('|__ train_loss: {:.4f}; valid_loss: {:.4f}'.format(
                train_loss, valid_loss))

            self.record_loss(epoch, train_loss, valid_loss)

    @timer
    @logger
    def run_epoch(self, epoch, data_loader):
        epoch_total_loss = 0.0 
        self.model.train()
        for (batch_idx, data) in enumerate(data_loader, 0):            
            img, age = data
            high_img, medium_img, low_img = img
            
            high_img = high_img.to(self.device)
            medium_img = medium_img.to(self.device)
            low_img = low_img.to(self.device)
            age = age.to(self.device)
            
            self.optimizer.zero_grad()
            pred_age, coarse_value, fine_value = self.model(high_img, medium_img, low_img)

            loss = self.criterion(pred_age, age)
            loss.backward()
            self.optimizer.step()
            
            epoch_total_loss += loss.item()
        
        return epoch_total_loss/len(data_loader)

    #@logger
    def evaluation(self, data_loader):
        losses = []
        accs = []
        self.model.eval()
        with torch.no_grad():
            for (batch_idx, data) in enumerate(data_loader, 0):
                img, age = data
                high_img, medium_img, low_img = img
            
                high_img = high_img.to(self.device)
                medium_img = medium_img.to(self.device)
                low_img = low_img.to(self.device)
                age = age.to(self.device)
                
                pred_age, coarse_value, fine_value = self.model(high_img, medium_img, low_img)
                
                loss = self.criterion(pred_age, age)
                losses.append(loss.item())
                
        return sum(losses)/len(losses)

class HSRTrainer(Trainer):
    def __init__(self, model, scheduler, criterion, optimizer, device='cpu', verbose=0, record_period=1):
        super(HSRTrainer, self).__init__(model, scheduler, criterion, optimizer, device, verbose, record_period)
    
    @logger
    def start_trainning_process(self, num_epoch, train_loader, valid_loader):
        for epoch in range(num_epoch):
            train_loss = self.run_epoch(epoch, train_loader)

            train_loss = self.evaluation(train_loader)
            valid_loss = self.evaluation(valid_loader)
                
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics=valid_loss)
            else:
                self.scheduler.step()
                
            print('|__ train_loss: {:.4f}; valid_loss: {:.4f}'.format(
                train_loss, valid_loss))

            self.record_loss(epoch, train_loss, valid_loss)

    @timer
    @logger
    def run_epoch(self, epoch, data_loader):
        epoch_total_loss = 0.0 
        self.model.train()
        for (batch_idx, data) in enumerate(data_loader, 0):            
            img, age = data
            
            img = img.to(self.device)
            age = age.to(self.device)
            
            self.optimizer.zero_grad()
            pred_age, coarse_value, fine_value = self.model(img)

            loss = self.criterion(pred_age, age)
            loss.backward()
            self.optimizer.step()
            
            epoch_total_loss += loss.item()
        
        return epoch_total_loss/len(data_loader)

    #@logger
    def evaluation(self, data_loader):
        losses = []
        accs = []
        self.model.eval()
        with torch.no_grad():
            for (batch_idx, data) in enumerate(data_loader, 0):
                img, age = data
                img = img.to(self.device)
                age = age.to(self.device)
                
                pred_age, coarse_value, fine_value = self.model(img)
                
                loss = self.criterion(pred_age, age)
                losses.append(loss.item())
                
        return sum(losses)/len(losses)

class C3AETrainer(Trainer):
    def __init__(self, model, scheduler, criterion, optimizer, device='cpu', verbose=0, record_period=1):
        super(C3AETrainer, self).__init__(model, scheduler, criterion, optimizer, device, verbose, record_period)
    
    @logger
    def start_trainning_process(self, num_epoch, train_loader, valid_loader):
        for epoch in range(num_epoch):
            train_loss = self.run_epoch(epoch, train_loader)
            
            train_loss = self.evaluation(train_loader)
            valid_loss = self.evaluation(valid_loader)
                
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics=valid_loss)
            else:
                self.scheduler.step()
                
            print('|__ train_loss: {:.4f}; valid_loss: {:.4f}'.format(
                train_loss, valid_loss))

            self.record_loss(epoch, train_loss, valid_loss)

    @timer
    @logger
    def run_epoch(self, epoch, data_loader):
        epoch_total_loss = 0.0 
        self.model.train()
        for (batch_idx, data) in enumerate(data_loader, 0):            
            img, age = data
            
            img = img.to(self.device)
            age = age.to(self.device)
            
            self.optimizer.zero_grad()
            pred_age = self.model(img)

            loss = self.criterion(pred_age, age)
            loss.backward()
            self.optimizer.step()
            
            epoch_total_loss += loss.item()
        
        return epoch_total_loss/len(data_loader)

    #@logger
    def evaluation(self, data_loader):
        losses = []
        accs = []
        self.model.eval()
        with torch.no_grad():
            for (batch_idx, data) in enumerate(data_loader, 0):
                img, age = data
                img = img.to(self.device)
                age = age.to(self.device)
                
                pred_age = self.model(img)
                
                loss = self.criterion(pred_age, age)
                losses.append(loss.item())
                
        return sum(losses)/len(losses)

class C3AEContextTrainer(Trainer):

    def __init__(self, model, scheduler, criterion, optimizer, device='cpu', verbose=0, record_period=1):
        super(C3AEContextTrainer, self).__init__(model, scheduler, criterion, optimizer, device, verbose, record_period)
    
    @logger
    def start_trainning_process(self, num_epoch, train_loader, valid_loader):
        for epoch in range(num_epoch):
            train_loss = self.run_epoch(epoch, train_loader)
            
            train_loss = self.evaluation(train_loader)
            valid_loss = self.evaluation(valid_loader)
                
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics=valid_loss)
            else:
                self.scheduler.step()
                
            print('|__ train_loss: {:.4f}; valid_loss: {:.4f}'.format(
                train_loss, valid_loss))

            self.record_loss(epoch, train_loss, valid_loss)

    @timer
    @logger
    def run_epoch(self, epoch, data_loader):
        epoch_total_loss = 0.0 
        epoch_kl_loss = 0.0
        self.model.train()
        for (batch_idx, data) in enumerate(data_loader, 0):            
            img, age, age_kl = data
            
            high_img, medium_img, low_img = img
            
            high_img = high_img.to(self.device)
            medium_img = medium_img.to(self.device)
            low_img = low_img.to(self.device)
            age = age.to(self.device)
            age_kl = age_kl.to(self.device)
            
            self.optimizer.zero_grad()
            pred_age, feat = self.model(high_img, medium_img, low_img)

            for param in self.model.f1.parameters():
                    kl_reg += torch.sum(torch.abs(param))
                    
            log_feat = F.log_softmax(feat.view_as(age_kl))

            kl_loss = F.kl_div(log_feat,age_kl,reduction='mean')
            reg_loss = self.criterion(pred_age, age) 
            loss = 10*kl_loss + reg_loss
            loss.backward()
            self.optimizer.step()
            
            epoch_total_loss += loss.item()
        
        return epoch_total_loss/len(data_loader)

    #@logger
    def evaluation(self, data_loader):
        losses = []
        accs = []
        self.model.eval()
        with torch.no_grad():
            for (batch_idx, data) in enumerate(data_loader, 0):
                img, age, age_kl = data
                high_img, medium_img, low_img = img
            
                high_img = high_img.to(self.device)
                medium_img = medium_img.to(self.device)
                low_img = low_img.to(self.device)
                age = age.to(self.device)
                
                pred_age, _ = self.model(high_img, medium_img, low_img)
                
                loss = self.criterion(pred_age, age)
                losses.append(loss.item())
                
        return sum(losses)/len(losses)