#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
from models.tools import quantizable_list,quantize_model
from util.nnUtils_for_quantification import pact_quantize,weight_quantize,quant_conv3x3,QuantConv2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152',]

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class QuantBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,k_bits = 8):
        super(QuantBasicBlock, self).__init__()

        self.k_bits = k_bits

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        ## Layer 1 ##
        self.bn1 = norm_layer(inplanes)
        self.relu1 = pact_quantize(self.k_bits)
        self.conv1 = quant_conv3x3(inplanes,planes,kernel_size=3,stride=stride,bias=False,k_bits=self.k_bits)

        ## Layer 2 ##
        self.bn2 = norm_layer(planes)
        self.relu2 = pact_quantize(self.k_bits)
        self.conv2 = quant_conv3x3(planes,planes,kernel_size=3,stride=1,bias=False,k_bits=self.k_bits)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.relu1(self.bn1(x))
        if self.downsample is not None:
            shortcut = self.downsample(out)
        else:
            shortcut = x
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)
        out += shortcut
        return out

class QuantBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,k_bits = 8):
        super(QuantBottleneck, self).__init__()
        self.k_bits = k_bits
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.bn1 = norm_layer(inplanes)
        self.relu1 = pact_quantize(inplanes)
        self.conv1 = quant_conv3x3(inplanes,width,kernel_size=1,stride=1, padding=0,dilation=1,bias=False,k_bits= self.k_bits)

        self.bn2 = norm_layer(width)
        self.relu2 = pact_quantize(self.k_bits)
        self.conv2 = quant_conv3x3(width,width,kernel_size=3,stride=stride,bias=False,k_bits=self.k_bits)
        
        self.bn3 = norm_layer(width)
        self.relu3 = pact_quantize(self.k_bits)
        self.conv3 = quant_conv3x3(width,planes * self.expansion,kernel_size=1,padding=0,dilation=1,stride=1,bias=False,k_bits = self.k_bits)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.relu1(self.bn1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)
        else:
            shortcut = x

        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)
        out = self.relu3(self.bn3(out))
        out = self.conv3(out)
        out += shortcut
  
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=200, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,k_bits = 8 ):
        super(ResNet, self).__init__()
        self.k_bits = k_bits
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dil
            # ated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        ## CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        ## END
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],k_bits=self.k_bits)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],k_bits=self.k_bits)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],k_bits=self.k_bits)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],k_bits=self.k_bits)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, QuantConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, QuantBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, QuantBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,k_bits = 8):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,k_bits= k_bits))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,k_bits = k_bits))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

def _resnet(arch, block, layers, init_k_bits,pre_k_bits,k_bits,num_layers,ratio,**kwargs):
    print(f'Building Architecture {arch}...')
    model = ResNet(block, layers, k_bits=init_k_bits,**kwargs)

    if type(k_bits) == list and len(k_bits) == (num_layers-2) * 2:
        quantize_bits = k_bits
        model,model_k_bits = quantize_model(model,quantize_bits,pre_k_bits=pre_k_bits,ratio=ratio)
    elif type(k_bits) == list and len(k_bits) != (num_layers-2) * 2:
        print(f"The Hyperparameters of k_bits is ERROR")
    elif type(k_bits) != list:
        quantize_bits = [k_bits for _ in range((num_layers-2) * 2)]
        model,model_k_bits = quantize_model(model,quantize_bits,pre_k_bits=pre_k_bits,ratio=ratio)
    else:
        print(f"ERROR")
        return 
    return model,model_k_bits

def resnet18(kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _resnet('resnet18',QuantBasicBlock,[2,2,2,2],init_k_bits= init_k_bits,\
                         pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio=ratio)

    return model,model_k_bits


def resnet34(kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _resnet('resnet34',QuantBasicBlock,[3, 4, 6, 3],init_k_bits= init_k_bits,\
                        pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio=ratio)

    return model,model_k_bits


def resnet50(kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _resnet('resnet50',QuantBottleneck,[3, 4, 6, 3],init_k_bits= init_k_bits,\
                        pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio = ratio)

    return model,model_k_bits


def resnet101(kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _resnet('resnet101',QuantBottleneck,[3, 4, 23, 3],init_k_bits= init_k_bits,\
                    pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio=ratio)

    return model,model_k_bits

def resnet152(kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _resnet('resnet152',QuantBottleneck,[3, 8, 36, 3],init_k_bits= init_k_bits,\
                    pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio=ratio)

    return model,model_k_bits

# def test():
#     k_bits = 8
#     num_layers = 18
#     pre_k_bits = 6
#     ratio = 1.0
#     kwargs = {'k_bits':k_bits,'num_layers':num_layers,'pre_k_bits':pre_k_bits,'ratio':ratio}
#     model,model_k_bits = resnet18(kwargs)
#     data = Variable(torch.randn(1,3,64,64))
#     outputs = model(data)
#     print(f'outputs {outputs}')
#     print(f'model_k_bits {model_k_bits}')

# test()