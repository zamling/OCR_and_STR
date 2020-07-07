import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from utils import utils
from modules import model_building
alphabet = 'abcdefghijklmnopqrstuvwxwz'
img_path = './data/demo.png'
net = model_building.ASTER(3,38,512,256,256,30)
demo = Image.open(img_path).convert('RGB')
trans = transforms.Compose([transforms.Resize((32,100)),transforms.ToTensor()])
img = trans(demo)
img = img.unsqueeze(0)
label = ['avaliable','avaliable','avaliable','avaliable']
convert = utils.ASTER_str2Int(alphabet,30)
x = convert.encoder(label)
print(x)
x = convert.decoder(x)
print(x)







