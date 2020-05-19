import torch.nn as nn
import torch

def hook_fn_forward(module, input, output):
    print(module) # 用于区分模块
    print('input shape', input[0].shape) # 首先打印出来
    print('output', output[0].shape)