from modules.model_building import crnn
from utils.utils import strLabelToInt
import torch
from PIL import Image
from torchvision import transforms



alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
model_root = './model'
img_path = './data/demo.png'

convert = strLabelToInt(alphabet)

model_path = model_root + '/netCRNN_9_8000.pth'

net = crnn(3,256,len(alphabet)+1)
net.load_state_dict(torch.load(model_path))
demo = Image.open(img_path)
trans = transforms.Compose([transforms.Resize((32,100)),transforms.ToTensor()])
img = trans(demo)
img = img.unsqueeze(0)
pred = net(img)
_,pred = pred.max(2)
length = torch.IntTensor([26])

raw_text = convert.decoder(pred,length)
pred_text = convert.decoder(pred,length,raw=False)
print('the word in pic is "Available"')
print('while the predicted result is {} from {}'.format(pred_text,raw_text))








