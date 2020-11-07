import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
from torch.autograd import Variable
from models.tools import quantizable_list,quantize_model
from util.nnUtils_for_quantification import quant_conv3x3,pact_quantize,weight_quantize,QuantConv2d
import pdb

__all__ = ['densenet40']

class DenseBasicBlock(nn.Module):
    def __init__(self, inplanes, filters,index,expansion=1, growthRate=12, dropRate=0,k_bits = 8):
        super(DenseBasicBlock, self).__init__()
        self.k_bits = k_bits

        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = pact_quantize(self.k_bits)
        self.conv = quant_conv3x3(filters,growthRate,kernel_size=3,stride=1,padding=1,bias=False,k_bits=self.k_bits)

        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out

class Transition(nn.Module):
    def __init__(self, inplanes, outplanes, filters, index,k_bits = 8):
        super(Transition, self).__init__()
        self.k_bits = k_bits
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = pact_quantize(self.k_bits)
        self.conv  = quant_conv3x3(filters,outplanes,kernel_size=1,stride=1,padding= 0,bias=False,k_bits=self.k_bits)
        
    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):

    def __init__(self, depth=40, block=DenseBasicBlock, 
        dropRate=0, num_classes=10, growthRate=12, compressionRate=2, filters=None, indexes=None,k_bits = 8):
        super(DenseNet, self).__init__()
        self.k_bits = k_bits
        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3 if 'DenseBasicBlock' in str(block) else (depth - 4) // 6
        transition = Transition

        if filters == None:
            filters = []
            start = growthRate*2
            for i in range(3):
                filters.append([start + growthRate*i for i in range(n+1)])
                start = (start + growthRate*n) // compressionRate
            filters = [item for sub_list in filters for item in sub_list]

            indexes = []
            for f in filters:
                indexes.append(np.arange(f))

        self.growthRate = growthRate
        self.dropRate = dropRate

        self.inplanes = growthRate * 2 
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, padding=1,bias=False)

        self.dense1 = self._make_denseblock(block, n, filters[0:n], indexes[0:n],self.k_bits)
        self.trans1 = self._make_transition(transition, compressionRate, filters[n], indexes[n],self.k_bits)
        self.dense2 = self._make_denseblock(block, n, filters[n+1:2*n+1], indexes[n+1:2*n+1],self.k_bits)
        self.trans2 = self._make_transition(transition, compressionRate, filters[2*n+1], indexes[2*n+1],self.k_bits)
        self.dense3 = self._make_denseblock(block, n, filters[2*n+2:3*n+2], indexes[2*n+2:3*n+2],self.k_bits)
        
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks, filters, indexes,k_bits):
        layers = []
        assert blocks == len(filters), 'Length of the filters parameter is not right.'
        assert blocks == len(indexes), 'Length of the indexes parameter is not right.'
        for i in range(blocks):
            layers.append(block(self.inplanes, filters=filters[i], index=indexes[i], growthRate=self.growthRate, dropRate=self.dropRate,k_bits=k_bits))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, transition, compressionRate, filters, index,k_bits):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return transition(inplanes, outplanes, filters, index,k_bits)

    def forward(self, x):
        x = self.conv1(x)

        x = self.dense1(x)
        x = self.trans1(x) 
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def _densenet(arch, block, layers, compressionRate, init_k_bits,pre_k_bits,k_bits,num_layers,ratio,**kwargs):
    print(f'Building Architecture {arch}...')
    model = DenseNet( layers, block,  compressionRate =compressionRate, k_bits=init_k_bits,**kwargs)

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

def densenet40(kwargs):

    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _densenet('densenet40',DenseBasicBlock,40,1,init_k_bits= init_k_bits,\
                    pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio=ratio)

    return model,model_k_bits

# def test():
#     k_bits = 6
#     num_layers = 40
#     pre_k_bits = 8
#     ratio = 0.6
#     kwargs = {'k_bits':k_bits,'num_layers':num_layers,'pre_k_bits':pre_k_bits,'ratio':ratio}
#     model,model_k_bits= densenet40(kwargs)
#     data = Variable(torch.randn(1,1,32,32))
#     outputs = model(data)
#     print(f'outputs {outputs}')
#     print(f'model_k_bits {model_k_bits}')

# test()