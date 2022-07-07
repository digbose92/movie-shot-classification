import torch.nn as nn 
import torch 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.autograd import Variable

class LSTM_model(nn.Module):
    def __init__(self, embedding_dim, n_hidden, n_classes, n_layers, batch_first=True):
        super().__init__()

        self.embedding_dim,self.n_hidden,self.n_classes, self.n_layers = embedding_dim, n_hidden, n_classes, n_layers
        self.batch_first=batch_first
        #embedding dimensions are individual frame embedding dimensions
        #hidden dimensions are dimensions of the LSTM
        #n_out 
        self.lstm = nn.LSTM(self.embedding_dim, self.n_hidden, self.n_layers, batch_first=self.batch_first) 
        self.out = nn.Linear(self.n_hidden, self.n_classes)
        
    def forward(self, embs, lengths):
        #print(embs.size())
        #seq should be (T,B,E)
        bs = embs.size(0) # batch size
        #print(self.batch_first)
        # self.h = self.init_hidden(bs) # initialize hidden state of LSTM
        #print(lengths)
        embs = pack_padded_sequence(embs, lengths=lengths, batch_first=self.batch_first) # unpad
        #print(embs)
        lstm_out, (hn,cn) = self.lstm(embs) # lstm returns hidden state of all timesteps as well as hidden state at last timestep
        
        lstm_out, lengths = pad_packed_sequence(lstm_out,batch_first=self.batch_first) # pad the sequence to the max length in the batch

        
        #source: https://blog.nelsonliu.me/2018/01/25/extracting-last-timestep-outputs-from-pytorch-rnns/
        idx = (torch.LongTensor(lengths) - 1).view(-1, 1).expand(len(lengths), lstm_out.size(2))
        time_dimension = 1 if self.batch_first else 0
        idx = idx.unsqueeze(time_dimension)
        if lstm_out.is_cuda:
                idx = idx.cuda(lstm_out.data.get_device())
        # Shape: (batch_size, rnn_hidden_dim)
        last_output = lstm_out.gather(
                time_dimension, Variable(idx)).squeeze(time_dimension)

        outp=self.out(last_output)
        #outp = self.out(hn[-1]) # self.h[-1] contains hidden state of last timesteps
        #variations to be tried here : adaptive max pool operations and concatenation
        # https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130
        return(outp)

# model=LSTM_model(768,512,5,2,True)
# embs=torch.randn((4,8,768))
# lengths=[7,6,2,1]
# val_op=model(embs,lengths)