import torch.nn as nn
import torch.nn.functional as F



class base_VGG(nn.Module):

    def __init__(self, imgH, nc, leakyRelu=False):
        super(base_VGG, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
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
        print(conv.size())
        #assert h == 1, "the height of conv must be 1"

        return conv

class debug_test(nn.Module):
    def __init__(self):
        super(debug_test,self).__init__()
        self.linear1 = nn.Linear(5,10)
        self.linear2 = nn.Linear(10,20)
        self.linear3 = nn.Linear(20,5)
    def forward(self,input):
        out1 = F.sigmoid(self.linear1(input))
        out2 = F.sigmoid(self.linear2(out1))
        out3 = self.linear3(out2)
        out3_cpu = out3.detach().cpu().numpy()
        return out3
