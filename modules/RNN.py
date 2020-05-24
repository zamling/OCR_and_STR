from torch import nn


'''
双向 LSTM
'''
class B_LSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(B_LSTM,self).__init__()
        self.rnn = nn.LSTM(nIn,nHidden,bidirectional=True)
        self.embedding = nn.Linear(nHidden*2,nOut)
    def forward(self, input):
        out,_ = self.rnn(input)
        T, b, h = out.size()
        out = out.view(T*b,h)
        pre = self.embedding(out)
        pre = pre.view(T,b,-1)

        return pre

