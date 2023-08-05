import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from nets.common_units import conv3x3_block, linear_block
import torch.nn.functional as F

# backend
class GlobalFeature(nn.Module):
    def __init__(self, kernel_size=30, compress=None):
        super(GlobalFeature, self).__init__()
        self.kernel_size = kernel_size
        self.compress = compress

        self.global_feat = nn.Sequential(OrderedDict([
            ('conv_bn_relu_1', conv3x3_block(3, self.kernel_size)),
            ('conv_bn_relu_2', conv3x3_block(self.kernel_size, self.kernel_size, compress=self.compress)),
            ('avg_pool_1', nn.AvgPool2d(2)),
            ('conv_bn_relu_3', conv3x3_block(self.kernel_size, self.kernel_size, compress=self.compress)),
            ('conv_bn_relu_4', conv3x3_block(self.kernel_size, self.kernel_size, compress=self.compress)),
            ('avg_pool_2', nn.AvgPool2d(2)),
            ('conv_bn_relu_5', conv3x3_block(self.kernel_size, self.kernel_size, compress=self.compress)),
            ('conv_bn_relu_6', conv3x3_block(self.kernel_size, self.kernel_size, compress=self.compress)),
            ('avg_pool_3', nn.AvgPool2d(2)),
        ]))
    
    def forward(self, x):
        feat = self.global_feat(x)
        return feat
    
# stage subnetwork
class StageWiseNetwork(nn.Module):
    def __init__(self, in_channels=30, out_channels=10, 
                 bin_array=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90], bin_weight_grad=True):
        super(StageWiseNetwork, self).__init__()
        self.bin_array = np.array(bin_array)
        self.bin_weight_grad = bin_weight_grad
        self.bin_size = len(self.bin_array)
        
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.avg_pool = nn.AvgPool2d(2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = linear_block(4*4*out_channels, self.bin_size, bias=False, activation=nn.ReLU())
        self.fc2 = linear_block(self.bin_size, 2*self.bin_size, bias=False, activation=nn.ReLU())
        self.age_dist = linear_block(2*self.bin_size, self.bin_size, bias=False, activation=nn.Softmax(dim=1))
        self.age_bin = linear_block(self.bin_size, 1, bias=False, activation=None)
        
        self.__init_age_bin__()
    
    def __init_age_bin__(self):
        with torch.no_grad():
            self.age_bin.linear.weight = nn.Parameter(torch.from_numpy(self.bin_array).float())
            if self.bin_weight_grad == False:
                for param in self.age_bin.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.avg_pool(x)
        flat_feat = self.flatten(x)
        
        x = self.fc1(self.dropout(flat_feat))
        x = self.fc2(x)
        age_distribution = self.age_dist(x)

        age = self.age_bin(age_distribution)
        return age


class HSRNet(nn.Module):
    def __init__(self, kernel_size=30, compress=None, out_channels=10, bin_weight_grad=True):
        super(HSRNet, self).__init__()
        # compress=None, return hsrnet architecture
        # compress='sep', return sephsrnet architecture
        # compress='bsep', return bsephsrnet architecture
        if compress == None:
            self.model_name = 'hsr'+str(kernel_size)+'-'+str(out_channels)
        elif compress == 'sep':
            self.model_name = 'sephsr'+str(kernel_size)+'-'+str(out_channels)
        elif compress == 'bsep':
            self.model_name = 'bsephsr'+str(kernel_size)+'-'+str(out_channels)
        else:
            raise ValueError('compress value error')
            
        self.kernel_size = kernel_size
        self.compress = compress
        self.out_channels = out_channels
        self.bin_weight_grad = bin_weight_grad
        
        self.global_feature = GlobalFeature(kernel_size=self.kernel_size, compress=self.compress)
        self.coarse_net = StageWiseNetwork(in_channels=self.kernel_size, out_channels=self.out_channels, 
                 bin_array=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90], bin_weight_grad=self.bin_weight_grad)
        self.fine_net = StageWiseNetwork(in_channels=self.kernel_size, out_channels=self.out_channels, 
                 bin_array=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], bin_weight_grad=self.bin_weight_grad)
    
    def forward(self, x):
        glob_feat = self.global_feature(x)
        coarse_value = self.coarse_net(glob_feat)
        fine_value = self.fine_net(glob_feat)
        
        pred_age = coarse_value + fine_value
        return pred_age, coarse_value, fine_value

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        print('Total parameters: {:,}\nTrainable parameters: {:,}\nNon-trainable parameters: {:,}'.format(
                total_params, trainable_params, non_trainable_params))
        return total_params, trainable_params, non_trainable_params
    
    def get_model_name(self):
        return self.model_name


class HSRNetContext(nn.Module):
    def __init__(self, kernel_size=30, compress=None, out_channels=10, bin_weight_grad=True, checkpoint_path=None):
        super(HSRNetContext, self).__init__()
        # compress=None, return hsrnet architecture
        # compress='sep', return sephsrnet architecture
        # compress='bsep', return bsephsrnet architecture
        if compress == None:
            self.model_name = 'hsr_context'+str(kernel_size)+'-'+str(out_channels)
        elif compress == 'sep':
            self.model_name = 'sephsr_context'+str(kernel_size)+'-'+str(out_channels)
        elif compress == 'bsep':
            self.model_name = 'bsephsr_context'+str(kernel_size)+'-'+str(out_channels)
        else:
            raise ValueError('compress value error')
            
        self.kernel_size = kernel_size
        self.compress = compress
        self.out_channels = out_channels
        self.bin_weight_grad = bin_weight_grad
        self.checkpoint_path = checkpoint_path
        
        self.global_feature = GlobalFeature(kernel_size=self.kernel_size, compress=self.compress)
        self.coarse_net = StageWiseNetwork(in_channels=3*self.kernel_size, out_channels=self.out_channels, 
                 bin_array=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90], bin_weight_grad=self.bin_weight_grad)
        self.fine_net = StageWiseNetwork(in_channels=3*self.kernel_size, out_channels=self.out_channels, 
                 bin_array=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], bin_weight_grad=self.bin_weight_grad)
    
        if self.checkpoint_path is not None:
            self._load_global_feature_weights()
    
    def _load_global_feature_weights(self):
        checkpoint_dict = torch.load(self.checkpoint_path)
        global_feature_state = self.global_feature.state_dict()

        for name, param in checkpoint_dict.items():
            if 'global_feature' in name:
                if isinstance(param, torch.nn.parameter.Parameter):
                    param = param.data
                name = name.replace('global_feature.','')
                global_feature_state[name].copy_(param)
            
    def forward(self, high_x, medium_x, low_x):
        high_glob_feat = self.global_feature(high_x)
        medium_glob_feat = self.global_feature(medium_x)
        low_glob_feat = self.global_feature(low_x)
        
        glob_feat = torch.cat([high_glob_feat, medium_glob_feat, low_glob_feat], 1)
        coarse_value = self.coarse_net(glob_feat)
        fine_value = self.fine_net(glob_feat)
        
        pred_age = coarse_value + fine_value
        return pred_age, coarse_value, fine_value

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        print('Total parameters: {:,}\nTrainable parameters: {:,}\nNon-trainable parameters: {:,}'.format(
                total_params, trainable_params, non_trainable_params))
        return total_params, trainable_params, non_trainable_params
    
    def get_model_name(self):
        return self.model_name


class C3AENet(nn.Module):
    def __init__(self):
        super(C3AENet, self).__init__()
        self.model_name = 'C3AE'
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        
        self.avg_pool = nn.AvgPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(num_features=32)
        
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.feat = nn.Conv2d(in_channels=32, out_channels=12, kernel_size=4)
        self.pred = nn.Conv2d(in_channels=12, out_channels=1, kernel_size=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.avg_pool(F.relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.avg_pool(F.relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.avg_pool(F.relu(self.bn3(x)))
        
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        
        x = self.dropout(self.conv5(x))
        feat = self.feat(x)
        softmax_feat = F.softmax(feat, dim=1)
        pred = self.pred(softmax_feat)
        pred_age = pred.view(pred.size(0))
        
        return pred_age
    
    def get_model_name(self):
        return self.model_name


class C3AENetContext(nn.Module):
    def __init__(self):
        super(C3AENetContext, self).__init__()
        self.model_name = 'C3AEContext'
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        
        self.avg_pool = nn.AvgPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(num_features=32)
        
        self.conv5 = nn.Conv2d(in_channels=3*32, out_channels=32, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.feat = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=4)
        self.pred = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=1)
    
    def forward(self, high_x, medium_x, low_x):
        
        # high resolution
        high_x = self.conv1(high_x)
        high_x = self.avg_pool(F.relu(self.bn1(high_x)))
        high_x = self.conv2(high_x)
        high_x = self.avg_pool(F.relu(self.bn2(high_x)))
        high_x = self.conv3(high_x)
        high_x = self.avg_pool(F.relu(self.bn3(high_x)))
        
        high_x = self.conv4(high_x)
        high_x = F.relu(self.bn4(high_x))
        
        # medium resolution
        medium_x = self.conv1(medium_x)
        medium_x = self.avg_pool(F.relu(self.bn1(medium_x)))
        medium_x = self.conv2(medium_x)
        medium_x = self.avg_pool(F.relu(self.bn2(medium_x)))
        medium_x = self.conv3(medium_x)
        medium_x = self.avg_pool(F.relu(self.bn3(medium_x)))
        
        medium_x = self.conv4(medium_x)
        medium_x = F.relu(self.bn4(medium_x))
        
        # low resolution
        low_x = self.conv1(low_x)
        low_x = self.avg_pool(F.relu(self.bn1(low_x)))
        low_x = self.conv2(low_x)
        low_x = self.avg_pool(F.relu(self.bn2(low_x)))
        low_x = self.conv3(low_x)
        low_x = self.avg_pool(F.relu(self.bn3(low_x)))
        
        low_x = self.conv4(low_x)
        low_x = F.relu(self.bn4(low_x))
        
        # concat three resolution facial images.
        glob_feat = torch.cat([high_x, medium_x, low_x], 1)
        
        x = self.dropout(self.conv5(glob_feat))
        feat = self.feat(x)
        softmax_feat = F.softmax(feat, dim=1)
        pred = self.pred(softmax_feat)
        pred_age = pred.view(pred.size(0))
        
        return pred_age, feat
    
    def get_model_name(self):
        return self.model_name
