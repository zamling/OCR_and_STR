import torch.nn as nn
import torch



class base_VGG(nn.Module):

    def __init__(self, nc, leakyRelu=False):
        super(base_VGG, self).__init__()
        #     0  1  2  3  4  5  6
        ks = [3, 3, 3, 3, 3, 3, 2]  #kernel size
        ps = [1, 1, 1, 1, 1, 1, 0]  #padding size
        ss = [1, 1, 1, 1, 1, 1, 1]  #stride size
        nm = [64, 128, 256, 256, 512, 512, 512] #channel size

        cnn = nn.Sequential()


        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]

            nOut = nm[i]

            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
                # BN的参数为上一层的channel数
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
        #kernel size (3,3), padding(1,1), stride(1,1)
        #imgH = (imgH-3+1*2)/1+1 = imgH
        #imgW = imgW
        #channel = 64
        convRelu(0)
        #imgH /= 2
        #imgW /= 2
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        #kernel size (3,3), padding(1,1), stride(1,1)
        #imgH = (imgH-3+1*2)/1+1 = imgH
        #imgW = imgW
        #channel = 128      
        convRelu(1)
        #imgH /= 2
        #imgW /= 2     --- shrinking 4 times
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        #kernel size (3,3), padding(1,1), stride(1,1)
        #imgH = (imgH-3+1*2)/1+1 = imgH
        #imgW = imgW
        #channel = 256 
        convRelu(2, True)
        #kernel size (3,3), padding(1,1), stride(1,1)
        #imgH = (imgH-3+1*2)/1+1 = imgH
        #imgW = imgW
        #channel = 256  ---shrinking 4 times
        convRelu(3)
        # pooling 2:
        #kernel (2,2), stride (2,1), padding(0,1)
#Fomulae : Lout = floor((Lin + 2*padding - dilation * (kernel_size - 1) - 1)/stride +1)
        #imgH = (imgH + 2*0 - 1*(2-1) -1)/2 +1 = (imgH - 2)/2 +1 = imgH/2 --H shrink 8 times
        #imgW = (imgW + 2*1 -1*(2-1)-1)/1 +1 = imgW + 1 --shrink 4 times , and plus 1

        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x32   #parameter: (kernel [2,2], stride [2,1], padding [0,1])
#H_out=floor( (H_in+2padding[0]-kernerl_size[0])/stride[0]+1 )
#W_out=floor( (W_in+2padding[1]-kernerl_size[1])/stride[1]+1 )
        convRelu(4, True)
        convRelu(5)
        #4,5 no change in shape, channel is 512

        # pooling 3:
        #imgH = imgH/2 ---shrink 16 times
        #imgW = imgW +1 --shrink 4 times , and plus 2
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x32
        # conv6
        # kernel (2,2), stride(1,1), padding(0,0)
        # imgH = (imgH-2)/1 + 1 = imgH - 1
        # imgW = (imgW-2)/1 + 1
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()  # b, 512, 1, 32
        # assert h == 1,"the height of conv must be 1"

        return conv

class ResnetUnit(nn.Module):
    def __init__(self,In,Out,stride):
        super(ResnetUnit,self).__init__()
        self.cov1 = nn.Conv2d(In,Out,kernel_size=1,stride=stride,bias=False)
        self.bn1 = nn.BatchNorm2d(Out)
        self.relu = nn.ReLU(inplace=True)
        self.cov2 = nn.Conv2d(Out,Out,kernel_size=3,padding=1,stride=1)
        self.bn2 = nn.BatchNorm2d(Out)
        self.shortcut = nn.Sequential()
        if stride != 1 or In != Out:
            self.shortcut.add_module('shortcut',nn.Conv2d(In,Out,kernel_size=1,stride=stride,bias=False))
            self.shortcut.add_module('Batchnorm',nn.BatchNorm2d(Out))
    def forward(self,input):
        conv1 = self.cov1(input)
        bn1 = self.bn1(conv1)
        re1 = self.relu(bn1)
        conv2 = self.cov2(re1)
        out = self.bn2(conv2)
        out += self.shortcut(input)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,nc):
        super(ResNet, self).__init__()
        self.in_plane = 32
        self.layer0 = nn.Sequential(nn.Conv2d(nc,32,kernel_size=3,stride=1,padding=1,bias=False),
                               nn.BatchNorm2d(32),
                               nn.ReLU(inplace=True))#[32,100]
        self.layer1 = self._make_layer(3,32,stride=[2,2])#[16,50]
        self.layer2 = self._make_layer(4,64,stride=[2,2])#[8,25]
        self.layer3 = self._make_layer(6,128,stride=[2,1])#[4,25]
        self.layer4 = self._make_layer(6,256,stride=[2,1])#[2,25]
        self.layer5 = self._make_layer(3,512,stride=[2,1])#[1,25]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)






    def _make_layer(self,num_blocks,planes,stride):
        strides = [stride]+[1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(ResnetUnit(self.in_plane,planes,strd))
            self.in_plane = planes
        return nn.Sequential(*layers)

    def forward(self,x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        _,_,h,_ = x5.size()
        assert h == 1, 'The output height must be 1,now is {}'.format(h)
        return x5

if __name__ == "__main__":
    net = base_VGG(3)
    for name, module in net.named_modules():
        print((name,module))







