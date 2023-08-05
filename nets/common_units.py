import torch
import torch.nn as nn
import collections
from collections import OrderedDict

def linear_block(in_features, out_features, bias=True, activation=None):
    return LinearActivation(in_features=in_features,
                            out_features=out_features,
                            bias=bias,
                            activation=activation)

def conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=True, compress=None):
    if compress==None:
        return ConvBNReLU(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=padding,
                        bias=bias)
    elif compress=='sep':
        return SeparableConvBNReLU(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=stride,
                                padding=padding,
                                bias=bias)
    elif compress=='bsep':
        return BSepConvBNReLU(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=stride,
                                padding=padding,
                                bias=bias)
    else:
        raise ValueError(f'compression method "{compress}" not exist.')


class LinearActivation(nn.Module):
    def __init__(self, in_features, out_features, bias, activation=None):
        super(LinearActivation, self).__init__()
        self.linear = nn.Linear(in_features=in_features,
                                out_features=out_features,
                                bias=bias)
        self.activation=activation
    
    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
            
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class SeparableConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(SeparableConvBNReLU, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels,
                                   out_channels=in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

# Blueprint Separable Convolution
class BSepConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(BSepConvBNReLU, self).__init__()
        self.pointwise = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   groups=1,
                                   bias=False)
        self.depthwise = nn.Conv2d(in_channels=out_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=out_channels,
                                   bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.pointwise(x)
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x