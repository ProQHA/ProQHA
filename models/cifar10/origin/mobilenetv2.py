# coding:utf-8

import torch
import torch.nn as nn
import math

__all__=['mobilenetv2']

def get_num_gen(gen):
    return sum(1 for x in gen)

def is_leaf(model):
    return get_num_gen(model.children()) == 0

def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def get_conv():
    return nn.Conv2d

#反转残差网络部分
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, type="conv2d", **kwargs):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        #断言函数，判断stride是否只有1，2两个值
        assert stride in [1, 2]
        #expand_ratio为拓展因子，可以将输入进行缩小，以至于减少输入的作用，通MobileNetV1的Response Multiple作用相似。
        hidden_dim = round(inp * expand_ratio)
        #使用残差连接方式进行连接
        self.use_res_connect = self.stride == 1 and inp == oup

        conv = get_conv()

        if expand_ratio == 1:
            # nn.Sequential：顺序容器，模块将按照它们在构造函数中传递的顺序添加到它中。
            # 或者，也可以传入模块的有序字典。
            self.conv = nn.Sequential(
                # hidden_dim:input channels
                # hidden_dim:output channels
                # 3:kernel_size
                # stride:stride
                # 1:padding
                # groups=hidden_dim:组
                # bias=False

                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv(hidden_dim, oup, 1, 1, 0, bias=False, **kwargs),
                nn.BatchNorm2d(oup),
            )
        else:

            self.conv = nn.Sequential(
                # pw
                conv(inp, hidden_dim, 1, 1, 0, bias=False, **kwargs),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv(hidden_dim, oup, 1, 1, 0, bias=False, **kwargs),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=10, input_size=32, width_mult=1., type="conv2d", **kwargs):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            #t:expansion actor
            #c:output channels
            #n:repeated times
            #s:stride

            ###ImageNet
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],

            # ###Cifar10
            # [1, 16, 1, 1],
            # [6, 24, 2, 1],
            # [6, 32, 3, 1],
            # [6, 64, 4, 2],
            # [6, 96, 3, 1],
            # [6, 160, 3, 2],
            # [6, 320, 1, 1],
        ]


        #'conv2d'
        self.type = type
        # building first layer
        # print(input_size)
        assert input_size % 32 == 0

        #宽度因子
        input_channel = int(input_channel * width_mult)

        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        ### Cifar10
        # self.features = [conv_bn(3, input_channel, 1)]
        ### ImageNet
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, type=type, **kwargs))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, type=type, **kwargs))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            #线性性
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #独立分布
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    #初始化权值

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #在卷积过程中，权重的初始化状态是与kernel_size相关的正态分布
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                #在批归一化过程中，权重矩阵的初始化状态是全1矩阵
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                #在线性化过程中，权重矩阵倍定义为确定均值和方差的正态分布
                #n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    #这一部分是什么作用？
    def init(self, prune=0, cluster=0):
        if self.type == "conv2d_pc":
            self.modify_forward(layername="Conv2d_pc", prune=prune, cluster=cluster)
        elif self.type == "conv2d_pcer":
            self.modify_forward(layername="Conv2d_pcer")
        elif self.type == "conv2d_pced":
            self.modify_forward(layername="Conv2d_pced")

    def modify_forward(self, layername="Conv2d", model=None, prune=0, cluster=0):
        if model is None:
            model = self
        for child in model.children():
            if is_leaf(child):
                if get_layer_info(child) in [layername]:
                    if layername == "Conv2d_pc":
                        child.prune(prune)
                        child.cluster(cluster)
                    else:
                        child.init()
                    print(child)
            else:
                self.modify_forward(layername=layername, model=child, prune=prune, cluster=cluster)

def mobilenetv2(pretrained=False,**kwargs):
    type='conv2d'
    model = MobileNetV2(type=type, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('mobilenetv2.pth.tar'))
    return model