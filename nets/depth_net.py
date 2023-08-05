import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from nets.common_units import conv3x3_block, linear_block
import torch.nn.functional as F

class HSRNetDepth31(nn.Module):
    def __init__(self, kernel_size=30, compress=False, out_channels=10, bin_weight_grad=True):
        super(HSRNetDepth31, self).__init__()
        if compress == False:
            self.model_name = 'hsr'+str(kernel_size)+'-depth31-'+str(out_channels)
        elif compress == True:
            self.model_name = 'sephsr'+str(kernel_size)+'-depth31-'+str(out_channels)
        else:
            raise ValueError('compress value error')
            
        self.kernel_size = kernel_size
        self.compress = compress
        self.out_channels = out_channels
        self.bin_weight_grad = bin_weight_grad
        
        self.depth_feature = DepthFeature(kernel_size=self.kernel_size, compress=self.compress)
        self.coarse_net = StageWiseNetwork(in_channels=self.kernel_size, out_channels=self.out_channels, 
                 bin_array=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90], bin_weight_grad=self.bin_weight_grad)
        self.fine_net = StageWiseNetwork(in_channels=self.kernel_size, out_channels=self.out_channels, 
                 bin_array=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], bin_weight_grad=self.bin_weight_grad)
    
    def forward(self, x):
        depth1_feat, depth2_feat, depth3_feat = self.depth_feature(x)
        coarse_value = self.coarse_net(depth3_feat)
        fine_value = self.fine_net(depth1_feat)
        
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

class HSRNetDepth32(nn.Module):
    def __init__(self, kernel_size=30, compress=False, out_channels=10, bin_weight_grad=True):
        super(HSRNetDepth32, self).__init__()
        if compress == False:
            self.model_name = 'hsr'+str(kernel_size)+'-depth32-'+str(out_channels)
        elif compress == True:
            self.model_name = 'sephsr'+str(kernel_size)+'-depth32-'+str(out_channels)
        else:
            raise ValueError('compress value error')
            
        self.kernel_size = kernel_size
        self.compress = compress
        self.out_channels = out_channels
        self.bin_weight_grad = bin_weight_grad
        
        self.depth_feature = DepthFeature(kernel_size=self.kernel_size, compress=self.compress)
        self.coarse_net = StageWiseNetwork(in_channels=self.kernel_size, out_channels=self.out_channels, 
                 bin_array=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90], bin_weight_grad=self.bin_weight_grad)
        self.fine_net = StageWiseNetwork(in_channels=self.kernel_size, out_channels=self.out_channels, 
                 bin_array=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], bin_weight_grad=self.bin_weight_grad)
    
    def forward(self, x):
        depth1_feat, depth2_feat, depth3_feat = self.depth_feature(x)
        coarse_value = self.coarse_net(depth3_feat)
        fine_value = self.fine_net(depth2_feat)
        
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

# backend
class DepthFeature(nn.Module):
    def __init__(self, kernel_size=30, compress=False):
        super(DepthFeature, self).__init__()
        self.kernel_size = kernel_size
        self.compress = compress

        self.conv_bn_relu_1 = conv3x3_block(3, self.kernel_size)
        self.conv_bn_relu_2 = conv3x3_block(self.kernel_size, self.kernel_size, compress=self.compress)
        self.avg_pool_1 = nn.AvgPool2d(2)
        self.conv_bn_relu_3 = conv3x3_block(self.kernel_size, self.kernel_size, compress=self.compress)
        self.conv_bn_relu_4 = conv3x3_block(self.kernel_size, self.kernel_size, compress=self.compress)
        self.avg_pool_2 = nn.AvgPool2d(2)
        self.conv_bn_relu_5 = conv3x3_block(self.kernel_size, self.kernel_size, compress=self.compress)
        self.conv_bn_relu_6 = conv3x3_block(self.kernel_size, self.kernel_size, compress=self.compress)
        self.avg_pool_3 = nn.AvgPool2d(2)
    
        self.depth1_adaptive_avg_pool = nn.AdaptiveAvgPool2d((8,8))
        self.depth2_adaptive_avg_pool = nn.AdaptiveAvgPool2d((8,8))
    def forward(self, x):
        x = self.conv_bn_relu_1(x)
        x = self.conv_bn_relu_2(x)
        depth1 = self.avg_pool_1(x)
        depth1_feat = self.depth1_adaptive_avg_pool(depth1)

        x = self.conv_bn_relu_3(depth1)
        x = self.conv_bn_relu_4(x)
        depth2 = self.avg_pool_2(x)
        depth2_feat = self.depth1_adaptive_avg_pool(depth2)

        x = self.conv_bn_relu_5(depth2)
        x = self.conv_bn_relu_6(x)
        depth3_feat = self.avg_pool_3(x)

        return depth1_feat, depth2_feat, depth3_feat
    
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
