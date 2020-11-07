import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
from models.tools import quantizable_list,quantize_model
from util.nnUtils_for_quantification import pact_quantize,weight_quantize,quant_conv3x3,\
                                            quant_linear,QuantConv2d,QuantLinear

__all__ = ['VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn','vgg19_bn', 'vgg19',]

class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True,k_bits = 8):
        super(VGG, self).__init__()
        self.k_bits=k_bits
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            # nn.Linear(512 * 7 * 7, 4096),
            # nn.ReLU(True),
            # # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # # nn.Dropout(),
            # nn.Linear(4096, num_classes),
            pact_quantize(self.k_bits),
            quant_linear(512 * 7 * 7, 4096,bias=True,k_bits=self.k_bits),
            # nn.Dropout(),
            pact_quantize(self.k_bits),
            quant_linear(4096, 4096,bias=True,k_bits=self.k_bits),
            # nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, QuantLinear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False,k_bits=8):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if in_channels == 3:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                relu = nn.ReLU(inplace=True)
            else:
                conv2d = quant_conv3x3(in_channels,v)
                relu = pact_quantize(k_bits)

            if batch_norm:
                layers += [nn.BatchNorm2d(in_channels),relu,conv2d]
            else:
                layers += [relu,conv2d]
            in_channels = v
    return nn.Sequential(*layers)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], ## (num_layers -4)*2:8--11
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], ## (num_layers -4)*2:10--13
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], ## (num_layers -4)*2:13--16
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], ## (num_layers -4)*2:16--19
}

def _vgg(arch, cfg, batch_norm,init_k_bits,pre_k_bits,k_bits,num_layers,ratio,**kwargs):
    print(f'Building Architecture {arch}...')
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm,k_bits=init_k_bits), **kwargs)

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

def vgg11(kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _vgg('vgg11','A', False,init_k_bits= init_k_bits,\
                        pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio=ratio)

    return model,model_k_bits

def vgg11_bn(kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _vgg('vgg11_bn','A', True,init_k_bits= init_k_bits,\
                        pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio=ratio)

    return model,model_k_bits


def vgg13(kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _vgg('vgg13','B', False,init_k_bits= init_k_bits,\
                        pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio=ratio)

    return model,model_k_bits

def vgg13_bn(kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _vgg('vgg13_bn','B', True,init_k_bits= init_k_bits,\
                        pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio=ratio)

    return model,model_k_bits

def vgg16(kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _vgg('vgg16','D', False,init_k_bits= init_k_bits,\
                        pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio=ratio)

    return model,model_k_bits

def vgg16_bn(kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _vgg('vgg16_bn','D', True,init_k_bits= init_k_bits,\
                        pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio=ratio)

    return model,model_k_bits

def vgg19(kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _vgg('vgg19','E', False,init_k_bits= init_k_bits,\
                        pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio=ratio)

    return model,model_k_bits

def vgg19_bn(kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    k_bits = kwargs['k_bits']
    num_layers = kwargs['num_layers']
    pre_k_bits = kwargs['pre_k_bits']
    ratio = kwargs['ratio']
    init_k_bits = 8

    model,model_k_bits = _vgg('vgg19_bn','E', True,init_k_bits= init_k_bits,\
                        pre_k_bits=pre_k_bits,k_bits=k_bits,num_layers=num_layers,ratio=ratio)

    return model,model_k_bits

# def test():
#     k_bits = 8
#     num_layers = 19
#     pre_k_bits = 6
#     ratio = 0.5
#     kwargs = {'k_bits':k_bits,'num_layers':num_layers,'pre_k_bits':pre_k_bits,'ratio':ratio}
#     model,model_k_bits = vgg19_bn(kwargs)
#     # model = model.cuda()
#     data = Variable(torch.randn(1,3,32,32))
#     outputs = model(data)
#     print(f'outputs {outputs}')
#     print(f'model_k_bits {model_k_bits}')

# test()