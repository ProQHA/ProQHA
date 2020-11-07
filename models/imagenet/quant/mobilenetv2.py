# coding:utf-8

import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
import torch
import torch.nn as nn
import math
from models.tools import quantizable_list,quantize_model
from util.nnUtils_for_quantification import pact_quantize,weight_quantize,QuantConv2d,quant_conv3x3
from torch.autograd import Variable
import pdb

__all__=['mobilenetv252']

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def quant_conv_bn(inp, oup, stride,k_bits = 8):
    return nn.Sequential(
        nn.BatchNorm2d(inp),
        pact_quantize(k_bits=k_bits),
        quant_conv3x3(inp,oup,kernel_size=3,stride=stride,padding=1,bias=False)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def quant_conv_1x1_bn(inp,oup,k_bits = 8):
    return nn.Sequential(
        nn.BatchNorm2d(inp),
        pact_quantize(k_bits=k_bits),
        quant_conv3x3(inp,oup,kernel_size=1,stride=1,padding=0,bias=False)
    )

def get_conv():
    return nn.Conv2d

#反转残差网络部分
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, k_bits = 8,**kwargs):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.k_bits = k_bits 

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                #pw
                nn.BatchNorm2d(hidden_dim),
                pact_quantize(self.k_bits),
                quant_conv3x3(hidden_dim,hidden_dim,3,1,stride,1,bias=False,groups=hidden_dim,k_bits=self.k_bits),

                # pw-linear
                nn.BatchNorm2d(hidden_dim),
                pact_quantize(self.k_bits),
                quant_conv3x3(hidden_dim,oup,1,0,1,1,bias=False) 
            )
        else:

            self.conv = nn.Sequential(
                # pw
                nn.BatchNorm2d(inp),
                pact_quantize(self.k_bits),
                quant_conv3x3(inp,hidden_dim,1,0,1,1,bias=False),

                # dw
                nn.BatchNorm2d(hidden_dim),
                pact_quantize(self.k_bits),
                quant_conv3x3(hidden_dim,hidden_dim,3,1,stride,1,bias=False,groups=hidden_dim),

                # pw-linear
                nn.BatchNorm2d(hidden_dim),
                pact_quantize(self.k_bits),
                quant_conv3x3(hidden_dim,oup,1,0,1,1,bias=False)
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1., k_bits = 8,**kwargs):
        super(MobileNetV2, self).__init__()
        self.k_bits = k_bits
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            #t:expansion actor
            #c:output channels
            #n:repeated times
            #s:stride

            ### ImageNet ###
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0

        input_channel = int(input_channel * width_mult)

        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        ### ImageNet ##
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t,k_bits=self.k_bits, **kwargs))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t,k_bits=self.k_bits,**kwargs))
                input_channel = output_channel

        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            # nn.Dropout(0.2),
            #线性性
            nn.Linear(self.last_channel, n_class),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def _mobilenet(arch,width_wult,init_k_bits,pre_k_bits,k_bits,num_layers,ratio):
    print(f'Building Architecture {arch}...')
    model = MobileNetV2(width_mult=width_wult,k_bits=init_k_bits)
    print(model)
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

def mobilenetv252(kwargs):
    
    width_mult = kwargs['width_mult']
    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _mobilenet('mobilenetv2',width_mult,init_k_bits,pre_k_bits,k_bits,num_layers,ratio)
    return model,model_k_bits

# def test():
#     k_bits = 6
#     num_layers = 52
#     pre_k_bits = 8
#     ratio =0.6
#     width_mult = 1.0
#     kwargs = {'k_bits':k_bits,'num_layers':num_layers,'pre_k_bits':pre_k_bits,'ratio':ratio,'width_mult':width_mult}
#     model,model_k_bits = mobilenetv2(kwargs)
#     data = Variable(torch.randn(1,3,224,224))
#     outputs = model(data)
#     print(f'outputs {outputs}')
#     print(f'model_k_bits {model_k_bits}')

# test()