from torch import nn
from .CNN import base_VGG
from .RNN import B_LSTM

class crnn(nn.Module):
    def __init__(self, nc, nh, nclass):
        super(crnn, self).__init__()
        self.cnn = base_VGG(nc)
        self.rnn1 = B_LSTM(512,nh,nh)
        self.rnn2 = B_LSTM(nh,nh,nclass)

    def forward(self,input):
        cnn_out = self.cnn(input)
        b,c,h,w = cnn_out.size()
        assert h==1,"the height must be 1"
        assert c==512,"the rnn input size must be 512(channel)"
        cnn_out = cnn_out.squeeze(2)
        rnn_input = cnn_out.permute(2,0,1)
        out1 = self.rnn1(rnn_input)
        out2 = self.rnn2(out1)
        out3 = nn.functional.log_softmax(out2,dim=2)
        return out3


