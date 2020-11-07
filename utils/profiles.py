import argparse
import sys
sys.path.append('../')
import models
from flops import *
from ast import literal_eval
from PIL import ImageFile
from params import print_model_parm_flops
from params import print_model_parm_nums
from models.imagenet.origin.resnet import resnet50
import torchvision.models as models

ImageFile.LOAD_TRUNCATED_IMAGES = True

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')

parser.add_argument('--model', '-a', metavar='MODEL', default='resnet_18',
                    choices=model_names, 
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help=
                    'type of tensor - e.g torch.cuda.HalfTensor')

parser.add_argument('--num_classes', default=10,help='the number of the data')

#MobileNet、IGCV
parser.add_argument('--width_mult',metavar='WIDTH_MULT',default=1.0,
                    help='the mult width')

#IGCV
parser.add_argument('--downsampling',default=8,help='Args of IGCV3')

#ResNet
parser.add_argument('--depth', default=18, help='resnet depth')

#channelnet
parser.add_argument('--vision', default=1, help='the vision of channelnet')

##edsr
parser.add_argument(
    '--n_resblocks', 
    type=int, 
    default=16,
    help='number of residual blocks'
    )

parser.add_argument(
    '--n_feats', 
    type=int, 
    default=64,
    help='number of feature maps'
    )

parser.add_argument(
    '--res_scale', 
    type=float, 
    default=1,
    help='residual scaling'
    )

parser.add_argument(
    '--rgb_range', 
    type=int, 
    default=255,
    help='maximum value of RGB'
    )

parser.add_argument(
    '--n_colors', 
    type=int, 
    default=3,
    help='number of color channels to use'
    )

parser.add_argument(
    '--scale', 
    type=str, 
    default='4',
    help='number of color channels to use'
    )


def main():

    global args
    args = parser.parse_args()
    
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))

    if args.dataset in ['cifar10', 'cifar100']:
        IMAGE_SIZE = 32
    else:
        IMAGE_SIZE =224

    model = models.resnet50()

    if args.model=='xception' :
        print_model_parm_nums(model)
        print_model_parm_flops(model, inputsize=(3, IMAGE_SIZE, IMAGE_SIZE))

    else:
        n_flops, n_params = measure_model(model,3,IMAGE_SIZE, IMAGE_SIZE)
        print('FLOPs: %.6fM, Params: %.6fM' % (n_flops / 1e6, n_params / 1e6))
        args.filename = "%s_%s_%s.txt" % \
                        (args.model, int(n_params), int(n_flops))

    del (model)

main()
