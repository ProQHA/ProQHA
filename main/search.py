import os
import time

from importlib import import_module
from tensorboardX import SummaryWriter
import numpy as np
import pdb
# from torchsummary import summary
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import StepLR
import sys
sys.path.append('../')
sys.path.append('../utils/')
import models
import utils
import utils.common as utils
from utils.options import args
from utils.get_flops import measure_model
from utils.data import get_dataset
from utils.preprocess import get_transform
from datetime import datetime
from ast import literal_eval
import torchvision
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from torchvision import datasets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from utils.data_cifar import get_cifar_iter_dali
from utils.data_imagenet import get_imagenet_iter_dali,get_imagenet_iter_torch
from scipy.stats import truncnorm
from lib.greed import architecture_search
# from util.adam import Adam
# from util.sgd import SGD

CIFAR10_TRAIN_LEN = 50000
CIFAR10_EVAL_LEN = 10000

def _make_dir(path):
    if not os.path.exists(path): os.makedirs(path)

# def load_check(checkpoint,model_s):
#     student_model_dict = model_s.state_dict()
#     teacher_pretrained_model = checkpoint
#     if teacher_pretrained_model:
#         teacher_model_key=[]
#         student_model_key=[]
#         for tkey in teacher_pretrained_model.keys():
#             teacher_model_key.append(tkey)
#         for skey in student_model_dict.keys():
#             student_model_key.append(skey)
            
#         for i in range(len(student_model_key)):
#             student_model_dict[student_model_key[i]] = teacher_pretrained_model[teacher_model_key[i]]
#     return student_model_dict

def load_check(checkpoint,model_s):
    student_model_dict = model_s.state_dict()
    teacher_pretrained_model = checkpoint
    if teacher_pretrained_model:
        teacher_model_key=[]
        student_model_key=[]
        for tkey in teacher_pretrained_model.keys():
            teacher_model_key.append(tkey)
        for skey in student_model_dict.keys():
            student_model_key.append(skey)
            
        for i in range(len(student_model_key)):
            student_model_dict[student_model_key[i]] = teacher_pretrained_model[teacher_model_key[i]]
    return student_model_dict

def main():
    start_epoch = 0
    best_prec1 = 0.0

    seed=np.random.randint(10000)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    if args.gpus is not None:
        device = torch.device("cuda:{}".format(args.gpus[0]))
        cudnn.benchmark = False
        cudnn.deterministic = True
        cudnn.enabled = True 
    else:
        device = torch.device("cpu")

    now = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    if args.mission is not None:
        if 'vgg' == args.arch and args.batchnorm:
            args.job_dir = f'{args.job_dir}/{args.dataset}/{args.arch}{args.num_layers}_bn/{args.mission}/{now}'
        elif 'resnet20' == args.arch:
            args.job_dir = f'{args.job_dir}/{args.dataset}/{args.arch}/{args.mission}/{now}'
        else:
            args.job_dir = f'{args.job_dir}/{args.dataset}/{args.arch}{args.num_layers}/{args.mission}/{now}'
    else:
        if 'vgg' == args.arch and args.batchnorm:
            args.job_dir = f'{args.job_dir}/{args.dataset}/{args.arch}{args.num_layers}_bn/{now}'
        else:
            args.job_dir = f'{args.job_dir}/{args.dataset}/{args.arch}{args.num_layers}/{now}'

    _make_dir(args.job_dir)
    ckpt = utils.checkpoint(args)
    print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
    utils.print_params(vars(args), print_logger.info)
    log_file=os.path.join(args.job_dir, 'search_log.csv')
    writer_train = SummaryWriter(args.job_dir +'/run/train')
    writer_test = SummaryWriter(args.job_dir+ '/run/test')

    ## hyperparameters settings ##
    n_layers = (args.num_layers - 2) * 2 
    unit_k_bits = int(args.k_bits)
    kbits_list = [unit_k_bits for i in range(n_layers)]
    print_logger.info(f'k_bits_list {kbits_list}')

    # Data loading
    print('=> Preparing data..')

    if args.dataset in ['cifar10', 'cifar100','mnist']:
        IMAGE_SIZE = 32
    else:
        IMAGE_SIZE = 224
    
    if args.dataset == 'imagenet':
        # train_loader = get_imagenet_iter_dali(type = 'train',image_dir=args.data_dir, batch_size=args.train_batch_size,num_threads=args.workers,crop=IMAGE_SIZE,device_id=0,num_gpus=1)
        # val_loader = get_imagenet_iter_dali(type='val', image_dir=args.data_dir, batch_size=args.eval_batch_size,num_threads=args.workers,crop=IMAGE_SIZE,device_id=0,num_gpus=1)
        train_data = get_imagenet_iter_torch(type='train',image_dir=args.base_data_dir,batch_size=args.train_batch_size,num_threads=args.workers,crop=IMAGE_SIZE,device_id=0,num_gpus=1)
        
    elif args.dataset == 'cifar10':
        train_transform, test_transform = utils._data_transforms_cifar10(cutout=args.cutout)
        train_data = torchvision.datasets.CIFAR10(args.data_dir, train=True, transform=train_transform, download=True)
        # test_data = torchvision.datasets.CIFAR10(args.data_dir,train=False, transform=test_transform, download=True)
        # train_loader = get_cifar_iter_dali(type='train', image_dir=args.data_dir, batch_size=args.train_batch_size,num_threads=args.workers)
        # val_loader = get_cifar_iter_dali(type='val', image_dir=args.data_dir, batch_size=args.eval_batch_size,num_threads=args.workers)

    # Create model
    # Create model
    print('=> Building model...')
    if args.dataset =='cifar10' or args.dataset == 'mnist':
        num_classes = 10
        train_data_length = 50000
        eval_data_length =10000
    elif args.dataset == 'imagenet':
        num_classes = 1000
        train_data_length = 50000
        eval_data_length =10000

    if args.arch == 'mobilenetv2':
        model_config = {'k_bits':kbits_list,'num_layers':args.num_layers,'pre_k_bits':args.pre_k_bits,'ratio':args.ratio,'width_mult':args.width_mult}
    else:
        model_config = {'k_bits':kbits_list,'num_layers':args.num_layers,'pre_k_bits':args.pre_k_bits,'ratio':args.ratio}

    if 'vgg' == args.arch and args.batchnorm:
        model,model_k_bits = import_module(f"models.{args.dataset}.{args.archtype}.{args.arch}").__dict__[f'{args.arch}{args.num_layers}_bn'](model_config)
    elif 'resnet20' == args.arch:
        model,model_k_bits = import_module(f"models.{args.dataset}.{args.archtype}.{args.arch}").__dict__[f'{args.arch}'](model_config)
    else:
        model,model_k_bits = import_module(f"models.{args.dataset}.{args.archtype}.{args.arch}").__dict__[f'{args.arch}{args.num_layers}'](model_config)

    model = model.to(device)

    print_logger.info(f'model_k_bits_list {model_k_bits}')

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    # Optionally resume from a checkpoint
    resume = args.resume
    
    if resume:
        print('=> Loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=device)
        state_dict = checkpoint['state_dict']
        start_epoch = checkpoint['epoch']
        pre_train_best_prec1 = checkpoint['best_prec1']
        model_check = load_check(state_dict,model)
        model.load_state_dict(model_check)
        print('Prec@1:',pre_train_best_prec1)
    else:
        checkpoint = model.state_dict()

    choose_model,k_bits = architecture_search(args=args,nn_model=model,device = device,checkpoint=checkpoint, \
                            step=args.step,criterion=criterion,train_data=train_data,train_batch_size=args.train_batch_size, \
                            eval_batch_size=args.eval_batch_size,train_data_length = train_data_length, \
                            eval_data_length = eval_data_length,clip_value=args.grad_clip,lam=args.lam,\
                            gpu_id = 0,print_logger = print_logger,ckpt = ckpt,log_file=log_file)


if __name__ == '__main__':
    main()