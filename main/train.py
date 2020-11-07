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
from utils.profiles import measure_model
from data.data import get_dataset
from data.preprocess import get_transform
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from torchvision import datasets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from data.data_cifar import get_cifar_iter_dali
from data.data_imagenet import get_imagenet_iter_dali
from scipy.stats import truncnorm

def _make_dir(path):
    if not os.path.exists(path): os.makedirs(path)

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
        # cudnn.deterministic = True
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
    elif args.dataset == 'tinyimagenet':
        IMAGE_SIZE = 64
    else:
        IMAGE_SIZE = 224

    if args.dataset == 'imagenet':
        train_loader = get_imagenet_iter_dali(type = 'train',image_dir=args.data_dir, batch_size=args.train_batch_size,num_threads=args.workers,crop=IMAGE_SIZE,device_id=0,num_gpus=1)
        val_loader = get_imagenet_iter_dali(type='val', image_dir=args.data_dir, batch_size=args.eval_batch_size,num_threads=args.workers,crop=IMAGE_SIZE,device_id=0,num_gpus=1)
    elif args.dataset == 'tinyimagenet':
        train_loader = get_imagenet_iter_dali(type = 'train',image_dir=args.data_dir, batch_size=args.train_batch_size,num_threads=args.workers,crop=IMAGE_SIZE,device_id=0,num_gpus=1)
        val_loader = get_imagenet_iter_dali(type='val', image_dir=args.data_dir, batch_size=args.eval_batch_size,num_threads=args.workers,crop=IMAGE_SIZE,device_id=0,num_gpus=1)
    elif args.dataset == 'cifar10':
        train_loader = get_cifar_iter_dali(type='train', image_dir=args.data_dir, batch_size=args.train_batch_size,num_threads=args.workers)
        val_loader = get_cifar_iter_dali(type='val', image_dir=args.data_dir, batch_size=args.eval_batch_size,num_threads=args.workers)

    # Create model
    print('=> Building model...')
    if args.dataset =='cifar10':
        num_classes = 10
        train_data_length = 50000
        eval_data_length =10000
    elif args.dataset == 'imagenet':
        num_classes = 1000
        train_data_length = 50000
        eval_data_length =10000

    # arch = args.arch
    # model = models.__dict__[arch]

    model_config = {'k_bits':kbits_list,'num_layers':args.num_layers,'pre_k_bits':args.pre_k_bits,'ratio':args.ratio}
    if args.arch == 'mobilenetv2':
        model_config = {'k_bits':kbits_list,'num_layers':args.num_layers,'pre_k_bits':args.pre_k_bits,'ratio':args.ratio,'width_mult':args.width_mult}
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
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[0.5 * args.train_epochs, 0.75 * args.train_epochs], gamma=0.1)
  
    # Optionally resume from a checkpoint
    resume = args.resume
    if resume:
        print('=> Loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=device)
        state_dict = checkpoint['state_dict']
        start_epoch = checkpoint['epoch']
        pre_train_best_prec1 = checkpoint['best_prec1']
        model_check = load_check(state_dict,model)
        pdb.set_trace()
        model.load_state_dict(model_check)
        print('Prec@1:',pre_train_best_prec1)

    if args.test_only:
        test_prec1 = test(args, device, val_loader, model, criterion, writer_test,print_logger,start_epoch )
        print('=> Test Prec@1: {:.2f}'.format(test_prec1))
        print(f'sample k_bits {kbits_list}')
        return

    for epoch in range(0, args.train_epochs):
        scheduler.step(epoch)
        train_loss, train_prec1 = train(args, device, train_loader, train_data_length, model, criterion, optimizer, writer_train, print_logger, epoch)
        test_prec1 = test(args, device, val_loader, eval_data_length, model, criterion, writer_test, print_logger, epoch)

        is_best = best_prec1 < test_prec1
        best_prec1 = max(test_prec1, best_prec1) 

        state = {
                'state_dict': model.state_dict(),
                'test_prec1': test_prec1, 
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1
            }
        ckpt.save_model(state, epoch + 1, is_best,mode='train')
        print_logger.info('==> BEST ACC {:.3f}'.format(best_prec1.item()))
        # print("=> Best accuracy {:.3f}".format(best_prec1.item()))

def train(args, device, loader_train, data_length, model, criterion, optimizer, writer_train, print_logger, epoch):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.train()
    
    # update learning rate
    for param_group in optimizer.param_groups:
        writer_train.add_scalar(
            'learning_rate', param_group['lr'], epoch
        )

    num_iterations = int(data_length/args.train_batch_size)

    for i, data in enumerate(loader_train):
        inputs = data[0]["data"].to(device)
        targets = data[0]["label"].squeeze().long().to(device)
        # compute output
        if 'quant' in args.arch and 'vgg' in args.arch:
            logits = model(inputs,epoch).to(device)
        else:
            logits = model(inputs).to(device)
        loss = criterion(logits, targets)
        
        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        losses.update(loss.item(), args.train_batch_size)

        for name,parameters in model.named_parameters():
            if 'alpha' in name:
                writer_train.add_scalar(name,parameters.data,num_iterations * epoch + i)
        writer_train.add_scalar(
            'train_top1', prec1[0], num_iterations * epoch + i
            )
        writer_train.add_scalar(
            'train_loss', loss.item(), num_iterations * epoch + i
            )
        top1.update(prec1[0], args.train_batch_size)
        top5.update(prec5[0], args.train_batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
 
        if args.clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # measure elapsed time
        if i % args.print_freq == 0:
            print_logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                epoch, i, num_iterations, loss=losses, top1=top1, top5=top5))
    loader_train.reset()
    return losses.avg, top1.avg

def test(args, device, loader_test, data_length, model, criterion, writer_test, print_logger, epoch=0):
    batch_time = utils.AverageMeter()
    total_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.eval()
    num_iterations = int(data_length/args.eval_batch_size)

    with torch.no_grad():
        print("=> Evaluating...")
        
        for i, data in enumerate(loader_test):
            inputs = data[0]["data"].to(device)
            targets = data[0]["label"].squeeze().long().to(device)

            # compute output
            if 'quant' in args.arch and 'vgg' in args.arch:
                logits = model(inputs,epoch).to(device)
            else:
                logits = model(inputs).to(device)
            loss = criterion(logits, targets)

            #measure accuracy and record loss
            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            losses.update(loss.item(), args.eval_batch_size)
            top1.update(prec1[0], args.eval_batch_size)
            top5.update(prec5[0], args.eval_batch_size)
        
        print_logger.info(
            'Epoch[{0}]({1}/{2}): '
            'Loss {loss.avg:.4f} '
            'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
            epoch, i, num_iterations, loss=losses, top1=top1, top5=top5))

    if not args.test_only:
        writer_test.add_scalar('test_top1', top1.avg, epoch)
    loader_test.reset()
    return top1.avg

if __name__ == '__main__':
    main() 




