import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from models.tools import quantizable_list,quantize_model
from util.nnUtils_for_quantification import pact_quantize,weight_quantize,quant_conv3x3,QuantConv2d

__all__ = ['resnet20', 'resnet56', 'resnet110']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, QuantConv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class QuantBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A',k_bits = 8):
        super(QuantBasicBlock, self).__init__()
        self.k_bits = k_bits

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = pact_quantize(self.k_bits)
        self.conv1= quant_conv3x3(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False,k_bits=self.k_bits)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = pact_quantize(self.k_bits)
        self.conv2 = quant_conv3x3(planes,planes,kernel_size=3,stride=1,padding=1,bias=False,k_bits=self.k_bits)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        ## Version 1##
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)
        out += shortcut
        return out

class QuantResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,k_bits = 8):
        super(QuantResNet, self).__init__()
        self.in_planes = 16
        self.k_bits = k_bits

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1,k_bits =self.k_bits)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2,k_bits =self.k_bits)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2,k_bits =self.k_bits)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride,k_bits = 8):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,k_bits=k_bits))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def _resnet(arch, block, layers, init_k_bits,pre_k_bits,k_bits,num_layers,ratio,**kwargs):
    print(f'Building Architecture {arch}...')
    model = QuantResNet(block, layers, k_bits=init_k_bits,**kwargs)

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

def resnet20(kwargs):
    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _resnet('resnet20',QuantBasicBlock,[3,3,3],init_k_bits= init_k_bits,\
                         pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio=ratio)
    return model,model_k_bits

def resnet32(kwargs):
    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _resnet('resnet32',QuantBasicBlock,[5, 5, 5],init_k_bits= init_k_bits,\
                         pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio=ratio)
    return model,model_k_bits

def resnet56(kwargs):
    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _resnet('resnet56',QuantBasicBlock,[9,9,9],init_k_bits= init_k_bits,\
                         pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio=ratio)
    return model,model_k_bits

def resnet110(kwargs):
    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _resnet('resnet110',QuantBasicBlock,[18, 18, 18],init_k_bits= init_k_bits,\
                         pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio=ratio)
    return model,model_k_bits

# def test():
#     k_bits = 4
#     num_layers = 110
#     pre_k_bits = 6
#     ratio = 0.5
#     kwargs = {'k_bits':k_bits,'num_layers':num_layers,'pre_k_bits':pre_k_bits,'ratio':ratio}
#     model,model_k_bits = resnet110(kwargs)
#     data = Variable(torch.randn(1,3,32,32))
#     outputs = model(data)
#     print(f'outputs {outputs}')
#     print(f'model_k_bits {model_k_bits}')

# test()