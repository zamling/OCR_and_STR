import cv2
from modules import CNN
import torch
from utils import utils

BATCH_SIZE = 10

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
net = CNN.debug_test().to(device)

#simualte the dataset
x = torch.normal(mean=torch.ones(1,5),std=0.1)

x = x.to(device)
out = net(x)
print(out)





# net = CNN.base_VGG(32,3)
# print(net)
# nets = net.named_modules()
# for name,module in nets:
#     module.register_forward_hook(utils.hook_fn_forward)



# img = cv2.imread('./data/demo.png')
# print('initial image shape: ',img.shape)
# img = cv2.resize(img,(100,32))#[H,W,C]
# #input: [B,C,H,W]
# img = torch.Tensor(img)
# img = img.permute(2,0,1)
# img = img.unsqueeze(0)
# output = net(img)
