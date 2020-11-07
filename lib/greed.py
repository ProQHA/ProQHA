#!/usr/bin/python3.7  
# -*- coding: utf-8 -*-

import time
import math
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
import random
import sys
sys.path.append('./')
sys.path.append('../')
from util.adam import Adam
from util.sgd import SGD
from utils.common import AverageMeter,accuracy,prGreen
from util.nnUtils_for_quantification import pact_quantize,weight_quantize,QuantConv2d,QuantLinear
from utils.data_cifar import get_cifar_iter_dali
from utils.data_imagenet import get_imagenet_iter_dali
import pdb
from torch.optim.lr_scheduler import MultiStepLR
from hessian_eigenthings import compute_hessian_eigenthings

CIFARTRAIN=50000
CIFARVAL=10000
QUANTIZABLE_TYPE_LIST = [pact_quantize,weight_quantize]
WEIGHT_EXTRACT = [QuantConv2d,QuantLinear]
CLIP_VALUE = 5.0
WIDTHSIZE = []
ORISIZE = 0.0

def write_row(filename, data, reset=False):
    with open(filename, 'w' if reset else 'a') as o:
        row = ''
        for i, c in enumerate(data):
            row += ('' if i == 0 else ',') + str(c)
        row += '\n'
        o.write(row)

def quantizable_list(model):
    global QUANTIZABLE_TYPE_LIST
    quantizable_idx = []
    quantizable_kbits = []
    layer_type_list = []
    for i, m in enumerate(model.modules()):
        if type(m) in QUANTIZABLE_TYPE_LIST:
            quantizable_idx.append(i)
            quantizable_kbits.append(m.k_bits)
            layer_type_list.append(m)
    return quantizable_idx,quantizable_kbits,layer_type_list

def quantizable_conv_list(model):
    global WEIGHT_EXTRACT
    quantizable_idx = []
    layer_type_list = []
    for i, m in enumerate(model.modules()):
        if type(m) in WEIGHT_EXTRACT:
            quantizable_idx.append(i)
            layer_type_list.append(m)
    return quantizable_idx,layer_type_list

def get_kbits_list(model):
    global QUANTIZABLE_TYPE_LIST
    kbits_idx = []
    layer_list = []
    for i, m in enumerate(model.modules()):
        if type(m) in QUANTIZABLE_TYPE_LIST:
            kbits_idx.append(m.k_bits)
            layer_list.append(m)
    return kbits_idx

def get_weight_size(model,quantizable_idx):
    # get the param size for each layers to prune, size expressed in number of params
    wsize_list = []
    for i, m in enumerate(model.modules()):
        if i in quantizable_idx:
            wsize_list.append(m.weight.data.numel())
    return wsize_list

def get_org_weight(model,wsize_list):
    org_weight = 0.
    org_weight += sum(wsize_list) * 32.
    return org_weight

def get_cur_weight(model,quantize_bits,wsize_list):
    cur_weight = 0.
    ## 此处需要获得整个网络weight的量化bits

    quantizable_kbits_list = []
    for num,i in enumerate(quantize_bits):
        if num%2 ==1:
            quantizable_kbits_list.append(i)

    assert len(quantizable_kbits_list) == len(wsize_list)
    for i, n_bit in enumerate(quantizable_kbits_list):
        cur_weight += n_bit * wsize_list[i]
    return cur_weight

def get_model_info(model):
    quantizable_idx,_ = quantizable_conv_list(model)
    wsize_list = get_weight_size(model,quantizable_idx)
    ori_size = get_org_weight(model,wsize_list)
    return wsize_list,ori_size

def get_model_size(model,quantize_bits,wsize_list,ori_size):
    cur_size = get_cur_weight(model,quantize_bits,wsize_list)
    cost = cur_size/ori_size
    return cost

def make_optimizer(model,args,optype,epochs):
    if optype == 'SGD':
        optim = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        if epochs == 1:
            sched = None
        else:
            sched = MultiStepLR(optim, milestones=[0.5 * epochs, 0.75 * epochs], gamma=0.1)
  
            # sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim,epochs,eta_min = 0)
        return optim,sched
    elif optype =='ADAM':
        optim = Adam(model.parameters(),lr= args.lr,betas= (args.beta1,args.beta2),eps=args.epsilon)
        if epochs ==1 :
            sched = None
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs,eta_min = 0)
        return optim,sched
    else:
        print("There are NO implemented now")

def quantize_model(model,quantizable_index,choose_index,quantize_bits,interval=1):
    quantize_layer_bit_dict = {n: b for n, b in zip(quantizable_index, quantize_bits)}
    model_k_bits = []
    
    for i, layer in enumerate(model.modules()):
        if i not in quantizable_index:
            continue
        n_bit = quantize_layer_bit_dict[i]
        if choose_index is not None:

            if i in choose_index:
                # step  = random.sample([1,2],1)
                n_bit -= interval
                if n_bit <= 2:
                    n_bit = 2
                layer.k_bits = n_bit
                model_k_bits.append(n_bit)
            else:
                layer.k_bits = n_bit
                model_k_bits.append(n_bit)
        else:
            layer.k_bits = n_bit
            model_k_bits.append(n_bit)
    return model,model_k_bits

def reward(self, acc, w_size_ratio=None):
    if w_size_ratio is not None:
        return (acc - self.org_acc + 1. / w_size_ratio) * 0.1
    return (acc - self.org_acc) * 0.1

def get_params(checkpoint):
    pre_params = []
    for pp in checkpoint:
        if ('running_mean' in pp) or  ('running_var' in pp) or ('num_batches_tracked' in pp):
            continue
        pre_params.append(checkpoint[pp])
    return pre_params

def architecture_search(args,nn_model,device,checkpoint,step,criterion,train_batch_size,eval_batch_size, train_data,\
                        train_data_length,eval_data_length,clip_value,lam=2, gpu_id=-1,log_file=None,print_logger=None,ckpt= None):
    
    print(f'Begin Searching')
    ## Loading Dataset ##
    wsize_list,ori_size = get_model_info(nn_model)
    inds = list(range(len(train_data)))
    eval_loader = torch.utils.data.DataLoader(train_data, train_batch_size, sampler=torch.utils.data.SubsetRandomSampler(inds[:train_batch_size]), num_workers=1)
    # train_data = get_imagenet_iter_torch(type='train',image_dir=args.base_data_dir,batch_size=args.train_batch_size,num_threads=args.workers,crop=IMAGE_SIZE,device_id=0,num_gpus=1)
    train_loader = get_imagenet_iter_dali(type='train', image_dir=args.data_dir, batch_size=args.train_batch_size,num_threads=args.workers,crop=224,device_id=0,num_gpus=1)
    val_loader = get_imagenet_iter_dali(type='val', image_dir=args.data_dir, batch_size=args.eval_batch_size,num_threads=args.workers,crop=224,device_id=0,num_gpus=1)

    if log_file is not None:
        write_row(log_file, ['step','step_time','model_size','hession_info','acc_info','cost','train_acc', 'test_acc','lr'] \
                            + ['kbits%d'%(i+1) for i in range((args.num_layers-2)*2)], reset=True)

    model = deepcopy(nn_model).to(device)

    cost = 1.0
    for s in range(step):
        t1 = time.time()
        hession_score = []
        acc_score = []
        kbits_list = []
        checkpoint = model.state_dict()

        nn_model = deepcopy(model)
        pre_params = get_params(checkpoint)

        quantizable_index,quantizable_kbits,_ = quantizable_list(nn_model)
        # print_logger.info('==> BEGIN STEP [{0}|{1}] KBITS {2}'.format(s+1,step,quantizable_kbits))
        for i in range(lam):
            nn_model.load_state_dict(checkpoint)
            optim,sched = make_optimizer(nn_model,args,'ADAM',1)
            # print_logger.info('==> BEST ACC {:.3f}'.format(best_prec1.item()))
            choose_index = random.sample(quantizable_index, int(len(quantizable_index)* args.ratio))

            quantize_model(nn_model,quantizable_index, choose_index,quantizable_kbits,interval = args.interval)
        
            nn_model,informance,top1 = finetune_one_batch(nn_model,pre_params,loader_train=eval_loader,data_length=train_data_length,\
                     device=device,criterion=criterion, optimizer=optim,scheduler=sched,\
                     print_freq=args.print_freq,print_logger=print_logger,step = s,\
                     batch_size=train_batch_size,epochs=1)

            # hvalue,hvector = hessian(nn_model,pre_params,loader_train=eval_loader,data_length=train_data_length,\
            #          device=device,criterion=criterion, optimizer=optim,scheduler=sched,\
            #          print_freq=args.print_freq,print_logger=print_logger,step = s,\
            #          batch_size=train_batch_size,epochs=1)

            # pdb.set_trace()
            test_top1 = test(model,loader_test=val_loader,data_length=eval_data_length,device=device,criterion=criterion,batch_size=eval_batch_size,print_logger=print_logger,step=s)
            # informance = max(hvalue)
            hession_score.append(informance)
            acc_score.append(test_top1)

            _,model_kbits,_ = quantizable_list(nn_model)
            cost = get_model_size(nn_model,model_kbits,wsize_list,ori_size)
            kbits_list.append(model_kbits)
            print_logger.info('==> SAMPLE STEP [{0}|{1}] SAMPLE [{2}|{3}] KBITS {4} Hession {infor} Acc {acc} COST {cost}'.format(s+1,step,i,lam,model_kbits,infor=informance,acc = test_top1,cost=cost))
        
        if args.info == 'hession':
            kbit_index = hession_score.index((min(hession_score)))
        else:
            kbit_index = acc_score.index((max(acc_score)))

        kbits = kbits_list[kbit_index]
        model,model_kbits_list = quantize_model(model,quantizable_index,None,kbits,interval= args.interval)
        optimizer,scheduler = make_optimizer(model,args,'ADAM',args.search_epochs)
        cost = get_model_size(model,model_kbits_list,wsize_list,ori_size)
        if args.info == 'hession':
            print_logger.info('==> CHOOSE STEP [{0}|{1}] KBITS {2} Info {infor} COST {cost}'.format(s+1,step,model_kbits_list,infor = min(hession_score),cost=cost))
        else:
            print_logger.info('==> CHOOSE STEP [{0}|{1}] KBITS {2} Info {infor} COST {cost}'.format(s+1,step,model_kbits_list,infor = max(acc_score),cost=cost))
        
        print_logger.info('==> Finetuning the choose model <==')
        model,train_acc = finetune(model,loader_train=train_loader,data_length=train_data_length,\
                     device=device,criterion=criterion, optimizer=optimizer,scheduler=scheduler,\
                     print_freq=args.print_freq,print_logger=print_logger,step = s,\
                     batch_size=train_batch_size,epochs=args.search_epochs)

        test_acc = test(model,loader_test=val_loader,data_length=eval_data_length,device=device,criterion=criterion,\
                    batch_size=eval_batch_size,print_logger=print_logger,step=s)

        print_logger.info('==> STEP [{0}|{1}] Prec@1 {prec:.2f}'.format(s,step,prec=test_acc))

        t2 = time.time()

        write_row(log_file, [s, t2-t1,ori_size,hession_score[kbit_index],acc_score[kbit_index],cost,train_acc,test_acc.item(),scheduler.get_lr()[0]]+ model_kbits_list)
        
        state = {
                'state_dict': model.state_dict(),
                'test_prec1': test_acc, 
                'cost':cost,
                'kbits':model_kbits_list,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'step': s
            }

        ckpt.save_model(state, s,False,mode='search')

    return model,model_kbits_list

def finetune_one_batch(model,pre_params,loader_train,data_length,device,criterion,optimizer,
             scheduler,print_freq, print_logger,step,batch_size,epochs=1,use_top5=False,verbose=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    best_acc = 0.
    informance = 0.0
    params = []
    
    model.train()
    end = time.time()
    t1 = time.time()

    for epoch in range(epochs):
        if scheduler is not None:
            scheduler.step(epoch)

        for batch_idx, data in enumerate(loader_train,0):
        # for i,(inputs,targets) in enumerate(loader_train,0):
            # pdb.set_trace()
            inputs,targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            # measure data loading time
            data_time.update(time.time() - end)

            optimizer.zero_grad()
            # compute output
            output = model(inputs)

            loss = criterion(output, targets)
            # compute gradient
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(),CLIP_VALUE)
            params = []
            optimizer.step()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, targets, topk=(1, 5))
            losses.update(loss.item(),batch_size)
            top1.update(prec1.item(), batch_size)
            top5.update(prec5.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print_logger.info(
                'Finetune One Batch Step [{0}]: '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f} '.format(
                step, loss=losses, top1=top1, top5=top5))

        for _, p in model.named_parameters():
            params.append(p)

        moment = optimizer.moment
        informance = [0.0 for i in range(len(moment))]

        suminfo = 0.0
        for i in range(len(moment)):
            informance[i] = moment[i] * torch.pow((pre_params[i]-params[i]),2)

        suminfo = 0.0
        for info in informance:
            suminfo += torch.sum(info).item()

        if use_top5:
            if top5.avg > best_acc:
                best_acc = top5.avg
        else:
            if top1.avg > best_acc:
                best_acc = top1.avg
        optimizer.moment = []
    return model,suminfo,top1.avg

def hessian(model,pre_params,loader_train,data_length,device,criterion,optimizer,
            scheduler,print_freq, print_logger,step,batch_size,epochs=1,use_top5=False,\
            verbose=True,num_eigenthings=20,mode ="power_iter",num_steps=500,max_samples= 512,momentum=0.0,full_dataset=True):
        # params = []
        eigenvals, eigenvecs = compute_hessian_eigenthings(
            model,
            loader_train,
            criterion,
            num_eigenthings,
            mode=mode,
            power_iter_steps=num_steps,
            max_samples=max_samples,
            momentum=momentum,
            full_dataset=full_dataset,
            use_gpu=True,
            )

        # print("Eigenvecs:")
        # print(eigenvecs)
        # print("Eigenvals:")
        # print(eigenvals)
        # # pdb.set_trace()
        return eigenvals,eigenvecs
        # for _, p in model.named_parameters():
        #     params.append(p)

        # moment = optimizer.moment
        # informance = [0.0 for i in range(len(moment))]

    #     suminfo = 0.0
    #     for i in range(len(moment)):
    #         informance[i] = moment[i] * torch.pow((pre_params[i]-params[i]),2)

    #     suminfo = 0.0
    #     for info in informance:
    #         suminfo += torch.sum(info).item()

    #     if use_top5:
    #         if top5.avg > best_acc:
    #             best_acc = top5.avg
    #     else:
    #         if top1.avg > best_acc:
    #             best_acc = top1.avg
    #     optimizer.moment = []
    # return model,suminfo,top1.avg


def finetune(model,loader_train,data_length,device,criterion,optimizer,scheduler,\
            print_freq, print_logger,step,batch_size,epochs=1,use_top5=False,verbose=True):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    best_acc = 0.

    # switch to train mode
    model.train()
    end = time.time()
    t1 = time.time()
    num_iterations = int(data_length/batch_size)

    for epoch in range(epochs):
        scheduler.step(epoch)

        for i, data in enumerate(loader_train):
            inputs = data[0]["data"].to(device)
            targets = data[0]["label"].squeeze().long().to(device)
            # measure data loading time
            data_time.update(time.time() - end)

            optimizer.zero_grad()
            # compute output
            output = model(inputs)
            loss = criterion(output, targets)
            # compute gradient
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(),CLIP_VALUE)

            optimizer.step()
            
            optimizer.moment = []

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, targets, topk=(1, 5))
            losses.update(loss.item(),batch_size)
            top1.update(prec1.item(), batch_size)
            top5.update(prec5.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print_logger.info(
                    'Finetune Step [{0}] Epoch [{1}|{2}] ({3}/{4}): '
                    'Loss {loss.avg:.4f} '
                    'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f} '.format(
                     step,epoch,epochs,i, num_iterations,loss=losses, top1=top1, top5=top5))

        if use_top5:
            if top5.avg > best_acc:
                best_acc = top5.avg
        else:
            if top1.avg > best_acc:
                best_acc = top1.avg
        loader_train.reset()
    return model,top1.avg

def validate(model,loader_val,data_length,device,criterion,
             batch_size,print_logger,step,use_top5=False,verbose=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    t1 = time.time()
    with torch.no_grad():
        # switch to evaluate mode
        model.eval()

        end = time.time()

        for i, data in enumerate(loader_val):
            inputs = data[0]["data"].to(device)
            targets = data[0]["label"].squeeze().long().to(device)

        # for i, (inputs, targets) in enumerate(loader_val, 1):
        #     inputs = inputs.to(device)
        #     targets = targets.to(device)

            # compute output
            output = model(inputs)
            loss = criterion(output, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, targets, topk=(1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(prec1.item(), batch_size)
            top5.update(prec5.item(),batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    t2 = time.time()

    print_logger.info(
            'Validate Step [{0}]: '
            'Loss {loss.avg:.4f} '
            'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f} '
            'Time {time}'.format(
                step, loss=losses, top1=top1, top5=top5,time = t2-t1))

    loader_val.reset()
    return top1.avg

def test(model,loader_test,data_length,device,criterion,
        batch_size,print_logger,step,use_top5=False,verbose=False):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        t1 = time.time()
        with torch.no_grad():
            # switch to evaluate mode
            model.eval()
            end = time.time()
            
            for i, data in enumerate(loader_test):
                inputs = data[0]["data"].to(device)
                targets = data[0]["label"].squeeze().long().to(device)

            # for i, (inputs, targets) in enumerate(loader_test, 1):
            #     inputs = inputs.to(device)
            #     targets = targets.to(device)

                # compute output
                output = model(inputs)
                loss = criterion(output, targets)

                #measure accuracy and record loss
                prec1, prec5 = accuracy(output, targets, topk=(1, 5))
                losses.update(loss.item(), batch_size)
                top1.update(prec1[0], batch_size)
                top5.update(prec5[0], batch_size)
            
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # plot progress
            
            # measure elapsed time
        t2 = time.time()
            
        print_logger.info(
            'Test Step [{0}]: '
            'Loss {loss.avg:.4f} '
            'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f} '
            'Time {time}'.format(
                step, loss=losses, top1=top1, top5=top5,time = t2-t1))

        loader_test.reset()
        return top1.avg

