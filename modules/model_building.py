from torch import nn
import torch
from .CNN import base_VGG,ResNet
from .RNN import B_LSTM, AttentionDecoder

class crnn(nn.Module):
    def __init__(self, nc, nh, nclass):
        super(crnn, self).__init__()
        self.cnn = base_VGG(nc)
        self.rnn1 = B_LSTM(512,nh,nh)
        self.rnn2 = B_LSTM(nh,nh,nclass)

    def forward(self,input):
        cnn_out = self.cnn(input)
        b,c,h,w = cnn_out.size()
        # assert h==1,"the height must be 1"
        # assert c==512,"the rnn input size must be 512(channel)"
        cnn_out = cnn_out.squeeze(2)
        rnn_input = cnn_out.permute(2,0,1)
        out1 = self.rnn1(rnn_input)
        out2 = self.rnn2(out1)
        out3 = nn.functional.log_softmax(out2,dim=2)
        return out3

class crnn_resnet(nn.Module):
    def __init__(self, nc, nh, nclass):
        super(crnn_resnet, self).__init__()
        self.cnn = ResNet(nc)
        self.rnn1 = B_LSTM(512,nh,nh)
        self.rnn2 = B_LSTM(nh,nh,nclass)

    def forward(self,input):
        cnn_out = self.cnn(input)
        b,c,h,w = cnn_out.shape
        assert h==1,"the height must be 1"
        assert c==512, "the rnn input size must be 512(channel),while the channel is {}".format(c)
        cnn_out = cnn_out.squeeze(2)
        rnn_input = cnn_out.permute(2,0,1)
        out1 = self.rnn1(rnn_input)
        out2 = self.rnn2(out1)
        out3 = nn.functional.log_softmax(out2,dim=2)
        return out3


class AttentionEncoder(nn.Module):
    def __init__(self,nc):
        super(AttentionEncoder,self).__init__()
        self.cnn = ResNet(nc)
        self.rnn = nn.LSTM(512,256,num_layers=2,bidirectional=True,batch_first=True)

    def forward(self,x):
        cnn_feats = self.cnn(x) # [b,512,1,25]
        cnn_feats = cnn_feats.squeeze(2)

        cnn_feats = cnn_feats.transpose(2,1) #[b,25,512]

        output,_ = self.rnn(cnn_feats)
        return output  #[b,25,512]



class ASTER(nn.Module):
    def __init__(self,nc,num_class,xDim,sDim,attDim,max_len_labels):
        super(ASTER,self).__init__()
        self.encoder = AttentionEncoder(nc)
        self.decoder = AttentionDecoder(num_class=num_class,xDim=xDim,sDim=sDim,attDim=attDim,max_len_labels=max_len_labels)

#    def forward(self,x):
#    def forward(self,x):
    def forward(self,img,labels,label_len):
        encoding = self.encoder(img)
        decoder_input = [encoding,labels,label_len]
        outputs = self.decoder(decoder_input)

        return outputs









