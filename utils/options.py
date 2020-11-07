import argparse
import os 

parser = argparse.ArgumentParser(description='Adversarial Network Compression')

## Environment Setting ##
parser.add_argument(
    '--gpus',
    type=int,
    nargs='+',
    default=[0],
    help='Select gpu to use')

## Dataset Setting ##
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=('cifar10', 'imagenet', 'mnist','cifar100', 'svhn','tinyimagenet'),
    help='Dataset to train')
parser.add_argument(
    '--data_dir',
    type=str,
    default='/userhome/memory_data/cifar10',
    help='The directory where the CIFAR-10 input data is stored.')
parser.add_argument(
    '--base_data_dir',
    type=str,
    default='/gdata/ImageNet2012',
    help='The directory where the CIFAR-10 input data is stored.')

parser.add_argument(
    '--workers',
    type=int,
    default=8,
    help='workers')

## Save File Setting ##
parser.add_argument(
    '--job_dir',
    type=str,
    help='The directory where the summaries will be stored.')
parser.add_argument(
    '--mission',
    type=str,
    help='Help you to record the mission.')
parser.add_argument(
    '--reset',
    action='store_true',
    help='Reset the directory?')

## Model Setting ##
parser.add_argument(
    '--resume', 
    type=str,
    help='load the model from the specified checkpoint')

parser.add_argument(
    '--arch',
    type=str,
    default='resnet',
    choices=('resnet','resnet20','vgg','densenet','mobilenetv2'),
    help='The architecture')

parser.add_argument(
    '--width_mult',
    type=float,
    default=1.0,
    help='The  witdth mult of the mobilenetv2')

parser.add_argument(
    '--batchnorm',
    type=str,
    default=True,
    help='Whether there is batchnorm in the model VGG')

parser.add_argument(
    '--archtype',
    type=str,
    default='quant',
    help='The architecture to quant')

parser.add_argument(
    '--num_layers',
    type=int,
    default=56,
    help='The number of resnet')

## Training, Validating, Testing, Finetuning Setting ## 
parser.add_argument(
    '--train_epochs', 
    type=int,
    default=300,
    help='The num of epochs to train.')
parser.add_argument(
    '--search_epochs', 
    type=int,
    default=30,
    help='The num of epochs to train.')
parser.add_argument(
    '--step',
    type=int,
    default=20
)
parser.add_argument(
    '--train_batch_size',
    type=int,
    default=128,
    help='Batch size for training.')
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=100,
    help='Batch size for validation.')

parser.add_argument(
    '--lr',
    type=float,
    default=0.1
)
parser.add_argument(
    '--weight_decay', 
    type=float,
    default=2e-4,
    help='The weight decay of loss.')

## SGD Setting ##
parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='Momentum for MomentumOptimizer.')

## Adam Setting ##
parser.add_argument(
    '--beta1',
    type=float,
    default=0.9
)
parser.add_argument(
    '--beta2',
    type=float,
    default=0.999
)
parser.add_argument(
    '--epsilon',
    type=float,
    default=10e-8
)

## Search Setting
parser.add_argument(
    '--info',
    type=str,
    default='hession',
    choices=('hession','acc'),
)
parser.add_argument(
    '--interval',
    type=int,
    default=1,
    choices=(1,2,4),
)
parser.add_argument(
    '--cost',
    type=float,
    default=0.25
)
parser.add_argument(
    '--k_bits',
    default=8
)
parser.add_argument(
    '--pre_k_bits',
    type=int,
    default=8
)
parser.add_argument(
    '--clip', 
    action="store_true", 
    default=False
    
)
parser.add_argument(
    '--ratio', 
    type= float, 
    default=0.5
)
parser.add_argument(
    '--lam',
    type=int,
    default=100
)
parser.add_argument(
    '--grad_clip',
    type=float,
    default=5.0,
)
parser.add_argument(
    '--cutout',
    type=bool,
    default=False,
)

## Result ##
parser.add_argument(
    '--print_freq', 
    type=int,
    default=100,
    help='The frequency to print loss.')
parser.add_argument(
    '--eval_freq', 
    type=int,
    default=1,
    help='The num of epochs to evaluate.')
parser.add_argument(
    '--test_only',
    default=False,
    action='store_true',
    help='test only')

args = parser.parse_args()

if args.resume is not None:
    if not os.path.isfile(args.resume):
        raise ValueError('No checkpoint found at {}'.format(args.resume))
