#coding:utf-8
import torch
import torchvision

import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

import numpy as np

def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name

def get_num_gen(gen):
    return sum(1 for x in gen)

def is_leaf(model):

    return get_num_gen(model.children()) == 0

def should_measure(x):
    return is_leaf(x)

def modify_forward(model,total):
    for child in model.children():
        if should_measure(child) :
            name = get_layer_info(child)
            if name=='Conv2d' or name=='Linear' or name=='BinarizeConv2d' or name=='BinarizeLinear':
                for param in child.parameters():
                    total.append(param.nelement())
        else:
            modify_forward(child,total)
    # print(total)
    return total

def print_model_parm_nums(model):
    total=[]
    count=0
    num=modify_forward(model,total)
    for i in num:
        count+=i
    print('  + Number of params: %.2fM' % (count / 1e6))


def print_model_parm_flops(model,inputsize):

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)


    multiply_adds = False
    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)


    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bilinear=[]
    def bilinear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        weight_ops_first = self.weight.nelement() * (2 if multiply_adds else 1)
        weight_ops_second = input[1].size(1) * output.size(0) * (2 if multiply_adds else 1)
        weight_ops = weight_ops_first + weight_ops_second
        bias_ops = self.bias.nelement()
        flops = batch_size * (weight_ops + bias_ops)
        list_bilinear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)


    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.Bilinear):
                net.register_forward_hook(bilinear_hook)
            return
        for c in childrens:
                foo(c)

    #resnet = models.alexnet()
    foo(model)
    input = Variable(torch.rand(inputsize).unsqueeze(0), requires_grad = True)
    out = model(input)


    total_flops = (sum(list_conv) + sum(list_linear) +sum(list_bilinear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))

    print('  + Number of FLOPs: %.2fM' % (total_flops / 1e6))
