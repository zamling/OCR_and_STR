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
from loss import sequenceCrossEntropyLoss as Loss
from PIL import Image

'''
annotation_train 7224612
annotation_test 891927 
annotation_val 802734 
'''

train_error= ['/2069/4/192_whittier_86389.jpg',
              '/2025/2/364_SNORTERS_72304.jpg',
              '/2013/2/370_refract_63890.jpg',
              '/1881/4/225_Marbling_46673.jpg',
              '/1863/4/223_Diligently_21672.jpg',
              '/1817/2/363_actuating_904.jpg',
              '/913/4/231_randoms_62372.jpg',
              '/869/4/234_TRIASSIC_80582.jpg',
              '/495/6/81_MIDYEAR_48332.jpg',
              '/368/4/232_friar_30876.jpg',
              '/275/6/96_hackle_34465.jpg',
              '/173/2/358_BURROWING_10395.jpg']

test_error = ['/2911/6/77_heretical_35885.jpg',
              '/2852/6/60_TOILSOME_79481.jpg',
              '/2749/6/101_Chided_13155.jpg']


BATCH_SIZE = 100
ROOT_DIR = './data/mnt/ramdisk/max/90kDICT32px'
TRAN_FILE_DIR = '/annotation_train.txt'
TEST_FILE_DIR = '/annotation_test.txt'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
IMAGE_CHANNEL = 3
NEPOCH = 10
DISPLAY = 2000
RECORD = 200
VALIDATION = 16000
SAVE = 35000
expr_dir = './model'

writer = SummaryWriter(comment='720-aster')
global_step = 0


cudnn.benchmark = True

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0),' is avaliable')
    device = torch.device('cuda:1')
else:
    print('using cpu actually')
    device = torch.device('cpu')







convert = utils.ASTER_str2Int(alphabet,30)
criterion = Loss.SequenceCrossEntropyLoss()
loss_avg_for_val = utils.Averager()
loss_avg_for_tra = utils.Averager()



net = model_building.ASTER(nc=IMAGE_CHANNEL,num_class=len(alphabet)+2,xDim=512,sDim=256,attDim=256,max_len_labels=30).to(device)


trans = transforms.Compose([transforms.Resize((32,100)),transforms.ToTensor()])

train_dataset = utils.OCR_dataset(ROOT_DIR,TRAN_FILE_DIR,num=7200000,errors=train_error,transform=trans)

train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,drop_last=True)

test_dataset = utils.OCR_dataset(ROOT_DIR,TEST_FILE_DIR,num=100000,errors=test_error,transform=trans)






'''
optimizer
'''
optimizer = torch.optim.Adadelta(net.parameters(),lr=1,weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[4,5],gamma=0.1)# adjust the optimizer's learning rate with epoch



#net = torch.nn.DataParallel(net,device_ids=range(torch.cuda.device_count()))


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
        Int_text = Int_text.to(device)
        Int_length = Int_length.to(device)
        img_feats = net.encoder(image)
        preds_ids, preds_scores = net.decoder.sample(img_feats)
        preds = net(image,Int_text,Int_length)
        cost = criterion(preds,Int_text,Int_length)
        loss_avg_for_val.add(cost)
        sim_preds = convert.decoder(preds_ids)
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
    Int_text = Int_text.to(device)
    Int_length = Int_length.to(device)
    preds = net(image,Int_text,Int_length) #size: T,b,h
    cost = criterion(preds,Int_text,Int_length)
    net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

for epoch in range(NEPOCH):
    scheduler.step(epoch)
    train_iter = iter(train_loader)
    i = 0
    tbar = tqdm(range(len(train_loader)))
    for i in tbar:
        tbar.set_description('trainning at epoch {}'.format(epoch+1))
        for p in net.parameters():
            p.requires_grad = True
        net.train()
        cost = trainBatch(net,criterion,optimizer)
        if i%RECORD == 0 and i != 0:
            writer.add_scalar('train losses',cost.item(),global_step=global_step)
            global_step += 1
        loss_avg_for_tra.add(cost)

        if i%DISPLAY == 0 and i != 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch+1, NEPOCH, i, len(train_loader), loss_avg_for_tra.val()))
            loss_avg_for_tra.reset()
        if i%VALIDATION == 0 and i != 0:
            Val(net,test_dataset,criterion)

        if i%SAVE == 0 and i != 0:
            torch.save(net.state_dict(),'{0}/720w_VGG_CRNN_{1}_{2}.pth'.format(expr_dir, epoch+1, i))


writer.close()