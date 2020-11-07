import sys
sys.path.append('../')
sys.path.append('../utils/')
import torch
import models
from utils.options import args
from importlib import import_module
from util.nnUtils_for_quantification import pact_quantize,weight_quantize,QuantConv2d,QuantLinear
QUANTIZABLE_TYPE_LIST = [pact_quantize,weight_quantize]
WEIGHT_EXTRACT = [QuantConv2d,QuantLinear]

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

def compute(model,k_bits):
    quantizable_index,quantizable_kbits,_ = quantizable_list(model)
    model,model_k_bits = quantize_model(model,quantizable_index,None,k_bits,)
    wsize_list,ori_size = get_model_info(model)
    cur_size = get_cur_weight(model,model_k_bits,wsize_list)

    return ori_size,cur_size

def main():
    ## hyperparameters settings ##
    n_layers = (args.num_layers - 2) * 2 

    kbits_list = [args.k_bits for i in range(n_layers)]

    model_config = {'k_bits':kbits_list,'num_layers':args.num_layers,'pre_k_bits':args.pre_k_bits,'ratio':args.ratio}
    if args.arch == 'mobilenetv2':
        model_config = {'k_bits':kbits_list,'num_layers':args.num_layers,'pre_k_bits':args.pre_k_bits,'ratio':args.ratio,'width_mult':args.width_mult}
    if 'vgg' == args.arch and args.batchnorm:
        model,model_k_bits = import_module(f"models.{args.dataset}.{args.archtype}.{args.arch}").__dict__[f'{args.arch}{args.num_layers}_bn'](model_config)

    else:
        model,model_k_bits = import_module(f"models.{args.dataset}.{args.archtype}.{args.arch}").__dict__[f'{args.arch}{args.num_layers}'](model_config)
    
    ori_size,cur_size = compute(model,kbits_list)
    print(f'model_k_bits_list {model_k_bits}')
    print(f'original {ori_size} current {cur_size}')
    print(f'computation ratio {cur_size/ori_size}')

if __name__ == '__main__':

    main()