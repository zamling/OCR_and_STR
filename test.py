from modules.model_building import crnn,crnn_resnet
from utils.utils import strLabelToInt
from tensorboardX import SummaryWriter
import torch
from PIL import Image
from torchvision import transforms



alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
model_root = './model'
img_path = 'world.PNG'
convert = strLabelToInt(alphabet)
model_dir = '/300w_VGG_CRNN_10_28000.pth'
model_path = model_root + model_dir

img_boxes = ['features.png','production.png','pytorch.png']

img = Image.open('hello.PNG')
net = crnn(3,256,len(alphabet)+1)
net.load_state_dict(torch.load(model_path))
for i in img_boxes:
    demo = Image.open(i).convert('RGB')
    trans = transforms.Compose([transforms.Resize((32,100)),transforms.ToTensor()])
    img = trans(demo)
    img = img.unsqueeze(0)
    pred = net(img)
    _,pred = pred.max(2)
    length = torch.IntTensor([26])

    raw_text = convert.decoder(pred,length)
    pred_text = convert.decoder(pred,length,raw=False)
    print('the word in pic is {}'.format(i.split('.')[0]))
    print('while the predicted result is {} from {}'.format(pred_text,raw_text))







