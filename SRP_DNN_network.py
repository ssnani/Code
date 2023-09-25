import torch
import torch.nn as nn

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pooling_kernel_size):
        super().__init__()
        self.basic_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True),

                                    nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True),

                                    nn.MaxPool2d(pooling_kernel_size),
                                )
        
    
    def forward(self, x):
        out = self.basic_block(x)
        return out


class SRPDNN(nn.Module):
    def __init__(self, num_freqs, bidirectional):
        super().__init__()
        self.freq = num_freqs
        #self.frms = num_frames

        self.rnn_hid_dim = self.freq 
        lin_inp = 2*self.freq if bidirectional else self.freq
        self.out_dim = 2*self.freq 

        #modified paper's architecture to maintain same #. of frames at the output
        self.blocklyrs = nn.Sequential(  BasicBlock( 4, 64, 3, 1, 1, (4,1)),
                                         BasicBlock(64, 64, 3, 1, 1, (2,1)),
                                         BasicBlock(64, 64, 3, 1, 1, (2,1)),
                                         BasicBlock(64, 64, 3, 1, 1, (2,1)),
                                         BasicBlock(64, 32, 3, 1, 1, (1,1))
                                        )

        self.rnn = nn.GRU(self.freq, self.rnn_hid_dim, 1, bias=True, batch_first=True, dropout=0.0, bidirectional=bidirectional)

        self.fc = nn.Sequential(nn.Linear(lin_inp, self.out_dim),
                                nn.Tanh()
                                )

    
    def forward(self, x):
        """
        x : (B, 4, F, T)
        pred : (B, T, 2*F)
        """
        feats = self.blocklyrs(x)
        batch_size, num_filters, freq, out_frm_rate = feats.shape
        _feats = torch.reshape(feats, (batch_size, num_filters*freq, out_frm_rate))
        
        rnn_out, _ = self.rnn(torch.permute(_feats,[0,2,1])) #torch.zeros(1,batch_size,self.freq).to(dtype = torch.float32, device = x.device)
        
        pred = self.fc(rnn_out)

        return pred



if __name__=="__main__":

    batch_size, F, N = 1, 160, 399
    input = torch.rand((batch_size, 10, 4,F,N))
    input = torch.reshape(input, (batch_size*10, 4, F, N))
    model = SRPDNN(F, True) #, N
    out = model(input)
    print(f"input: {input.shape}, out: {out.shape}")
    breakpoint()

    for layer_name, param in model.named_parameters():
        print(f"{layer_name}: {param.grad}")