from __future__ import absolute_import
from pathlib import Path
import datetime
import shutil
import torch.nn as nn
import logging
import os
import torch.nn.functional as F
import pdb
import torch
import functools
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from torch.autograd import  Variable
from PIL import Image
import torchvision.transforms as transforms
plt.rcParams['axes.unicode_minus'] = False

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_cifar10(cutout=False, cutout_length=16):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if cutout:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform

def drop_path(X, drop_prob):
    assert 0. <= drop_prob and drop_prob < 1.
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.FloatTensor(X.size(0), 1, 1, 1).bernoulli_(keep_prob)
        if X.is_cuda:
            mask = mask.cuda(X.get_device())
        X = X*mask/keep_prob
    return X

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class checkpoint():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        self.args = args
        self.ckpt_dir = self.args.job_dir+'/checkpoint'
        self.run_dir = self.args.job_dir + '/run'

        if args.reset:
            os.system('rm -rf ' + args.job_dir)

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.args.job_dir)
        _make_dir(self.ckpt_dir)
        _make_dir(self.run_dir)
        
        config_dir = self.args.job_dir + '/config.txt'
        with open(config_dir, 'w') as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    
    def save_model(self, state, step, is_best, mode='train',name='model_best'):
        if 'train' == mode:
            if is_best:
                save_path = '{}/model_best.pt'.format(self.ckpt_dir)
                print('=> Saving model to {}'.format(save_path))
                torch.save(state, save_path)
        else:
            if is_best:
                save_path = '{}/model_best.pt'.format(self.ckpt_dir)
                print('=> Saving model to {}'.format(save_path))
                torch.save(state, save_path)
            else:
                save_path = f'{self.ckpt_dir}/{step}.pt'
                print('=> Saving model to {}'.format(save_path))
                torch.save(state, save_path)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    logsoftmax = nn.LogSoftmax()
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))

def wrapped_partial(func, *args, **kwargs):
  partial_func = functools.partial(func, *args, **kwargs)
  functools.update_wrapper(partial_func, func)
  return partial_func

def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('kd')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

def print_params(config, prtf=print):
    prtf("")
    prtf("Parameters:")
    for attr, value in sorted(config.items()):
        prtf("{}={}".format(attr.upper(), value))
    prtf("")

def as_markdown(config):
    """ Return configs as markdown format """
    text = "|name|value|  \n|-|-|  \n"
    for attr, value in sorted(config.items()):
        text += "|{}|{}|  \n".format(attr, value)

    return text

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0),-1))

def at_loss(x,y):
    return (at(x)-at(y)).pow(2).mean()

def _make_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def adjust_learning_rate(lr,optimizer, epoch):
    '''
        Sets the learning rate to the initial LR decayed by a factor of 10
        every 30 epochs
    '''

    lr = lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def fun(weight):

    standard = torch.zeros_like(weight)
    weight = torch.ge(weight,standard).float().cuda()
    return weight

# def Gba(unit_value,M=1e5,h=1):
#     unit = unit_value.detach()
#     G = fun(unit)
#     S = h * (M * unit - torch.floor(M * unit))/M
#     # pdb.set_trace()
#     unit = G + S
#     return unit

def Gba(unit_value,M=1e5,h=1):
    unit = unit_value.detach()
    G = fun(unit)
    S = h * (torch.ceil(M * unit) - (M * unit))/M
    unit = G + S
    return unit

def GFunction(unit):
    unit_detach = unit.detach()
    G = fun(unit_detach)
    ones = torch.sum(G.view(-1) == 1).item()
    num = torch.numel(G)
    out = ones/num
    return out

def Flop(operation,mask1,ratio=0.5,alpha=0.2,beta=0.6):
    remain = torch.tensor([443018.0]).float().cuda()
    Ratio = torch.tensor([ratio]).float().cuda()
    P=torch.tensor([0.0]).float().cuda()
    S=torch.tensor([0.0]).float().cuda()
    length = len(mask1)
    mask1_unit = mask1
    ## hlplus1_1, hlplus1_2, kernel, inp1, oup1, inp2, oup2, sum
    ##     0        1           2      3     4     5     6    7
    value = []

    for i in range(length):

        FlopP = operation[0][i] ** 2 * (operation[2] ** 2 * operation[3][i] * torch.mul(operation[4][i], GFunction(mask1_unit[i]))) +\
                 operation[1][i] ** 2 * (operation[2] ** 2 * torch.mul(operation[5][i],GFunction(mask1_unit[i])) * operation[6][i])
        FlopS = operation[0][i] ** 2 * (operation[2] ** 2 * operation[3][i] * operation[4][i]) + \
                operation[1][i] ** 2 * (operation[2] ** 2 * operation[5][i] * operation[6][i])

        FlopP = FlopP.float().cuda()
        FlopS = FlopS.float().cuda()
        P = P + FlopP
        S = S + FlopS

        value.append(FlopP/FlopS)

    value =torch.stack(tuple(value))

    fraction = torch.norm(Ratio-value,p=2).float().cuda()

    return fraction,P/1e6,S/1e6,(P/S)

def SpatialAvgPool(fm):
    data = torch.sum(fm,1)
    # data = torch.mean(fm,1)
    data =torch.unsqueeze(data,1)
    return data


def hist(writer,data,name,epoch):

    x = data.view(-1).detach()
    x = x.cpu().numpy()
    num_bins = 50
    # pdb.set_trace()
    plt.figure()
    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(x, num_bins, density=1)

    # add a 'best fit' line
    ax.set_xlabel('Number')
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of MASK.UNIT')
    ax.grid(True)
    # ax.title(str(epoch)+'_'+str(name))
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()

    writer.add_figure(name,fig,epoch)


class Logger(object):
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
from torch.autograd import Variable

def to_numpy(var):
    # return var.cpu().data.numpy()
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

def sample_from_truncated_normal_distribution(lower, upper, mu, sigma, size=1):
    from scipy import stats
    return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)

# logging
def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))

