import random
import sys
sys.path.append('../')
from util.nnUtils_for_quantification import pact_quantize,weight_quantize
from torch.autograd import Variable
import pdb
QUANTIZABLE_TYPE_LIST = [pact_quantize,weight_quantize]

def quantizable_list(model):
    global QUANTIZABLE_TYPE_LIST
    quantizable_idx = []
    layer_type_list = []
    for i, m in enumerate(model.modules()):
        if type(m) in QUANTIZABLE_TYPE_LIST:
            quantizable_idx.append(i)
            layer_type_list.append(m)
    return quantizable_idx,layer_type_list


def quantize_model(model,quantize_bits,ratio = 0.5,pre_k_bits = 8):
    quantize_index,_ = quantizable_list(model)
    assert len(quantize_index) == len(quantize_bits)
    choose_quantize = random.sample(quantize_index, int(len(quantize_index)*ratio))
    quantize_layer_bit_dict = {n: b for n, b in zip(quantize_index, quantize_bits)}
    model_k_bits = []

    ## Random Half Sample ## 
    if ratio < 1.0:
        for i, layer in enumerate(model.modules()):
            if i not in quantize_index:
                continue
            n_bit = quantize_layer_bit_dict[i]
            if i in choose_quantize:
                layer.k_bits = n_bit
                model_k_bits.append(n_bit)
            else:
                layer.k_bits = pre_k_bits
                model_k_bits.append(pre_k_bits)
    else:
    ## Setting ##
        for i, layer in enumerate(model.modules()):
            if i not in quantize_index:
                continue
            n_bit = quantize_layer_bit_dict[i]
            layer.k_bits = n_bit
        model_k_bits = quantize_bits   

    return model,model_k_bits
