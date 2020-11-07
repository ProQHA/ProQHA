#!/usr/bin/python3.6  
# -*- coding: utf-8 -*-

from torch.autograd import Function as F
import torch.nn as nn
import torch
import math
import numpy as np
import pdb
import collections
from itertools import repeat
import sys

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)

def shift_sigmoid(x,slope=1.,constant=-2.5):
  return (1.0 / (1.0 + torch.exp(-(slope * x - constant))))

def gradient_accelerate(x,M = 1e5):
  return (x * M - torch.floor(x * M) - 0.5) / M

def write_row(filename, data, reset=False):
    with open(filename, 'w' if reset else 'a') as o:
        row = ''
        for i, c in enumerate(data):
            row += (str(c)+',' if i == 0 else str(c)+'\n')
        o.write(row)

def statistics_weight(abs_loss,norm_loss,path):
    data = []
    # data.append(k_bits)
    data.append(float((abs_loss).cpu().numpy()))
    data.append(float((norm_loss).cpu().numpy()))
    write_row(path,data,reset=False)
    return 

def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = float(2 ** k - 1)
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply

class weight_quantize(nn.Module):
  def __init__(self, k_bits):
    super(weight_quantize, self).__init__()
    self.k_bits = k_bits
    self.uniform_q = None
    
  def forward(self, x):
    self.uniform_q = uniform_quantize(self.k_bits)
    if self.k_bits == 32:
      weight_q = x
    elif self.k_bits == 1:
      E = torch.mean(torch.abs(x)).detach()
      weight_q = self.uniform_q(x / E) * E
    else:
      weight = torch.tanh(x)
      weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
      weight_q = 2 * self.uniform_q(weight) - 1
    return weight_q

def identity_gradient_function(**kargs):
  class identity_quant_function(torch.autograd.Function):
    ## Forward ##
    @staticmethod
    def forward(ctx, input):
        out = torch.round(input)
        return out

    ## Backward ##
    @staticmethod
    def backward(ctx, grad_output):
      return grad_output
  return identity_quant_function().apply

class pact_quantize(nn.Module):
  def __init__(self, k_bits):
    super(pact_quantize, self).__init__()
    self.k_bits = k_bits
    self.uniform_q = identity_gradient_function()
    self.alpha = nn.Parameter(torch.Tensor(1))
    self.reset_parameter()
    
  def reset_parameter(self):
    nn.init.constant_(self.alpha,10.0)

  def forward(self, input):
    if self.k_bits == 32:
      input = 0.5 * (torch.abs(input)-torch.abs(input-self.alpha)+self.alpha)
      activation_q = input
    else:
      input = 0.5 * (torch.abs(input)-torch.abs(input-self.alpha)+self.alpha)
      activation = input * (2 ** self.k_bits-1)/self.alpha
      middle_activation = self.uniform_q(activation) 
      activation_q = middle_activation * self.alpha / (2**self.k_bits-1)
    return activation_q

class QuantConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=False,k_bits=8):
        super(QuantConv2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        if self.groups != 1:
          self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels//groups,kernel_size,kernel_size))
        else:
          self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,kernel_size,kernel_size))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias
        if self.bias_flag:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias',None)
        self.k_bits = k_bits
        self.quantize_weight = weight_quantize(k_bits = self.k_bits)

    def forward(self, input, order=None):
        weight_q = self.quantize_weight(self.weight)
        
        return nn.functional.conv2d(input, weight_q, self.bias, self.stride,
                        padding=self.padding, dilation=self.dilation,groups= self.groups)

class QuantLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False,k_bits=8):
        super(QuantLinear, self).__init__()
        self.k_bits = k_bits
        self.quantize_fn = weight_quantize(k_bits=self.k_bits)
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels))
        self.bias_flag = bias
        if self.bias_flag:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias',None)

    def forward(self, input):
        weight_q = self.quantize_fn(self.weight)
        return nn.functional.linear(input, weight_q, self.bias)

def conv3x3(in_channels, out_channels,kernel_size=3,stride=1,padding =1,bias= True):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)

def quant_conv3x3(in_channels, out_channels,kernel_size=3,padding = 1,stride=1,dilation =1,bias = True,groups=1,k_bits=8):
    Conv= QuantConv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=kernel_size, stride = stride,padding=padding,dilation=dilation,bias = bias,groups=groups,k_bits=k_bits)
    return Conv

def quant_linear(in_channels,out_channels,bias = True,k_bits=8):
    Linear = QuantLinear(in_channels,out_channels,bias=bias,k_bits=k_bits)
    return Linear
