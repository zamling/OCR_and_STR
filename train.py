import cv2
import torch
import torch.nn as nn
from utils import utils
from modules import model_building
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
from PIL import Image

'''
annotation_train 7224612
annotation_test 891927 
annotation_val 802734 
'''

BATCH_SIZE = 64
ROOT_DIR = './data/mnt/ramdisk/max/90kDICT32px'
TRAN_FILE_DIR = '/annotation_train.txt'
TEST_FILE_DIR = '/annotation_test.txt'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
IMAGE_CHANNEL = 3
HIDDERN_SIZE_LSTM = 256
NEPOCH = 10
DISPLAY = 1000
RECORD = 200
VALIDATION = 8000
SAVE = 50000
expr_dir = './model'

writer = SummaryWriter(comment='crnn-resnet')
global_step = 0


cudnn.benchmark = True

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0),' is avaliable')
    device = torch.device('cuda:1')
else:
    print('using cpu actually')
    device = torch.device('cpu')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)






convert = utils.strLabelToInt(alphabet)
criterion = nn.CTCLoss()
loss_avg_for_val = utils.Averager()
loss_avg_for_tra = utils.Averager()





trans = transforms.Compose([transforms.Resize((32,100)),transforms.ToTensor()])

train_dataset = utils.OCR_dataset(ROOT_DIR,TRAN_FILE_DIR,num=7200000,transform=trans)

train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,drop_last=True)

test_dataset = utils.OCR_dataset(ROOT_DIR,TEST_FILE_DIR,num=10000,transform=trans)

net = model_building.crnn_resnet(IMAGE_CHANNEL,HIDDERN_SIZE_LSTM,len(alphabet)+1).to(device)
net.apply((weights_init))

demo_dir = './data/demo.png'



# if opt.adam:
#     optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr,
#                            betas=(opt.beta1, 0.999))
# elif opt.adadelta:
#     optimizer = torch.optim.Adadelta(net.parameters())
# else:
optimizer = torch.optim.Adam(net.parameters(),lr=0.0001,betas=(0.5,0.999))

# nets = net.named_modules()
# for name,module in nets:
#     module.register_forward_hook(utils.hook_fn_forward)

# net = torch.nn.DataParallel(net,device_ids=range(torch.cuda.device_count()))
criterion = criterion.to(device)

def Val(net,dataset,criterion,max_iter=100):
    print("Start Validation")

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,drop_last=True)
    val_iter = iter(data_loader)
    n_correct = 0
    max_iter = min(len(data_loader),max_iter)
    pbar = tqdm(range(max_iter))
    for i in pbar:
        pbar.set_description('running evaluation')
        data = val_iter.next()
        cpu_image, cpu_text = data
        image = cpu_image.to(device)
        Int_text,Int_length = convert.encoder(cpu_text)
        preds = net(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * BATCH_SIZE)) #batch*[seq_len]
        cost = criterion(preds,Int_text,preds_size,Int_length)/BATCH_SIZE
        loss_avg_for_val.add(cost)
        _, preds = preds.max(2)

        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = convert.decoder(preds,preds_size,raw=False)
        if i%5 == 0 and i != 0:
            print('\n',"the predicted text is {}, while the real text is {}".format(sim_preds[0], cpu_text[0]))
        for pred, target in zip(sim_preds,cpu_text):
            if pred == target.lower():
                n_correct += 1
    accuracy = n_correct / float(max_iter * BATCH_SIZE)

    print('Test loss: %f, accuray: %f' % (loss_avg_for_val.val(), accuracy))
    loss_avg_for_val.reset()

def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_image, cpu_text = data
    image = cpu_image.to(device)
    Int_text,Int_length = convert.encoder(cpu_text)
    assert len(Int_text) == Int_length.sum(), 'the encoded text length is not equal to variable length '
    preds = net(image) #size: T,b,h
    preds_size = Variable(torch.IntTensor([preds.size(0)] * BATCH_SIZE))  # batch*seq_len
    cost = criterion(preds, Int_text, preds_size, Int_length) / BATCH_SIZE
    net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

for epoch in range(NEPOCH):
    train_iter = iter(train_loader)
    i = 0
    tbar = tqdm(range(len(train_loader)))
    for i in tbar:
        tbar.set_description('running trainning')
        for p in net.parameters():
            p.requires_grad = True
        net.train()
        cost = trainBatch(net,criterion,optimizer)
        if i%RECORD == 0 and i != 0:
            writer.add_scalar('train losses',cost.item(),global_step=global_step)
            global_step += 1
        loss_avg_for_tra.add(cost)
        i += 1
        if i%DISPLAY == 0 and i != 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch+1, NEPOCH, i, len(train_loader), loss_avg_for_tra.val()))
            loss_avg_for_tra.reset()
        if i%VALIDATION == 0 and i != 0:
            Val(net,test_dataset,criterion)

        if i%SAVE == 0 and i != 0:
            torch.save(net.state_dict(),'{0}/720w_Resnet_CRNN_{1}_{2}.pth'.format(expr_dir, epoch+1, i))


writer.close()




















