import torch.nn as nn
import torch
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import Dataset
import os
from PIL import Image
import collections
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2




def hook_fn_forward(module, input, output):
    print(module) # 用于区分模块
    print('input shape', input[0].shape) # 首先打印出来
    print('output', output[0].shape)


class OCR_dataset(Dataset):
    def __init__(self,root_dir,file_dir,num,errors,transform = None):
        super(OCR_dataset,self).__init__()
        self.num = num
        self.root_dir = root_dir
        self.file = file_dir
        self.datas = []
        self.transform = transform
        ab_file_dir = root_dir + file_dir
        with open(ab_file_dir,'r') as f:
            dirty_data = f.readlines()
        pbar = tqdm(range(num))
        for i in pbar:
            pbar.set_description('loading dataset from{}'.format(file_dir))
            pri_data = dirty_data[i].strip().split()[0][1:]
            if pri_data not in errors:
                self.datas.append(pri_data)

    def __len__(self):
        # assert len(self.datas) == self.num,"the loading data's length has a error"
        return len(self.datas)

    def __getitem__(self, item):
        image_name = self.datas[item]
        image_dir = self.root_dir + image_name
        img = Image.open(image_dir)
        if self.transform is not None:
            img = self.transform(img)
        label = image_name.split('/')[-1].split('.')[0].split('_')[-2]
        return (img,label)





class ASTER_str2Int (object):
    def __init__(self,alphabet,max_len,is_ignore = True):
        self._is_ignore = is_ignore
        self.max_len = max_len
        if self._is_ignore:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '$' + '-'
        self.dict = {}
        for i,char in enumerate(self.alphabet):
            self.dict[char] = i
    def encoder(self,words):
        texts = []
        text_len = []
        for i in words:
            text = [self.dict[char.lower() if self._is_ignore else char] for char in i]
            text.append(self.dict['$'])
            text_len.append(len(text))
            res_text = [self.dict['-'] for i in range(self.max_len)]
            res_text[:len(text)] = text
            texts.append(res_text)
        return torch.IntTensor(texts), torch.IntTensor(text_len)

    def decoder(self,words):
        texts = []
        for i in words:
            print(i)
            res_str = ''
            for j in i:
                if j == self.dict['$']:
                    texts.append(res_str)
                    break
                else:
                    res_str += self.alphabet[j]
        return texts










class strLabelToInt(object):
    def __init__(self, alphabet, is_ignore = True):
        self._is_ignore = is_ignore
        if self._is_ignore:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'
        self.dict = {}
        for i,char in enumerate(alphabet):
            self.dict[char] = i + 1 # the '-' = 0

    def encoder(self,words):
        if isinstance(words,str):
            text = [self.dict[char.lower() if self._is_ignore else char] for char in words] #text is a list in Int
            length = len(text)
        elif isinstance(words,collections.Iterable):
            length = [len(s) for s in words]
            text = ''.join(words)
            text ,_ = self.encoder(text)
        return (torch.IntTensor(text),torch.IntTensor(length))

    def decoder(self,text,length,raw = True):
        if length.numel() == 1:
            length = length.item()
            assert length == text.numel(),"text has the length {}, while the claimed length is {}".format(text.numel(),length)
            if raw:
                str_text = ''.join([self.alphabet[i-1] for i in text]) # 0-1 = -1 最后一个是'-'
                return str_text
            else:
                char_list = []
                for i in range(length):
                    if text[i] != 0 and (not(i>0 and text[i-1] == text[i])):
                        char_list.append(self.alphabet[text[i]-1])
                return ''.join(char_list)
        else:
            assert text.numel() == length.sum(),"the batch text has the length {}, while the claimed length is {}".format(text.numel(),length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decoder(text[index:index+l],length[i],raw = raw)
                )
                index += l
            return texts









class Averager(object):

    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res











