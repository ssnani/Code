import torch
import torch.nn as nn
import torch.nn.functional as F

#from utils.utils import numParams
OLD_IMPLEMENTATION = False

class GLSTM(nn.Module):
    def __init__(self, hidden_size=1024, groups=4, bidirectional=True):
        super(GLSTM, self).__init__()
   
        hidden_size_t = hidden_size // groups

        hid_out = hidden_size_t//2 if bidirectional else hidden_size_t
        self.lstm_list1 = nn.ModuleList([nn.LSTM(hidden_size_t, hid_out, 1, batch_first=True, bidirectional=bidirectional) for i in range(groups)])
        self.lstm_list2 = nn.ModuleList([nn.LSTM(hidden_size_t, hid_out, 1, batch_first=True, bidirectional=bidirectional) for i in range(groups)])

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
     
        self.groups = groups
     
    def forward(self, x):
        out = x
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1).contiguous()
    
        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.stack([self.lstm_list1[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.ln1(out)
    
        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.cat([self.lstm_list2[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = self.ln2(out)
    
        out = out.view(out.size(0), out.size(1), x.size(1), -1).contiguous()
        out = out.transpose(1, 2).contiguous()
      
        return out


class GluConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(GluConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
   
        self.sigmoid = nn.Sigmoid()
   
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


class GluConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(GluConvTranspose2d, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.sigmoid(self.conv2(x))
        out = out1 * out2
        return out


class DenseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, grate):
        super(DenseConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, grate, (1,3), padding=(0,1))
        self.conv2 = nn.Conv2d(in_channels+grate, grate, (1,3), padding=(0,1))
        self.conv3 = nn.Conv2d(in_channels+2*grate, grate, (1,3), padding=(0,1))
        self.conv4 = nn.Conv2d(in_channels+3*grate, grate, (1,3), padding=(0,1))
        self.conv5 = GluConv2d(in_channels+4*grate, out_channels, kernel_size, padding=padding, stride=stride)
 
        self.bn1 = nn.BatchNorm2d(grate)
        self.bn2 = nn.BatchNorm2d(grate)
        self.bn3 = nn.BatchNorm2d(grate)
        self.bn4 = nn.BatchNorm2d(grate)
        
        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()
        self.elu3 = nn.ELU()
        self.elu4 = nn.ELU()
  
    def forward(self, x):
        out = x
        out1 = self.elu1(self.bn1(self.conv1(out)))
        out = torch.cat([x, out1], dim=1)
        out2 = self.elu2(self.bn2(self.conv2(out)))
        out = torch.cat([x, out1, out2], dim=1)
        out3 = self.elu3(self.bn3(self.conv3(out)))
        out = torch.cat([x, out1, out2, out3], dim=1)
        out4 = self.elu4(self.bn4(self.conv4(out)))
        out = torch.cat([x, out1, out2, out3, out4], dim=1)
        out5 = self.conv5(out)
    
        out = out5
    
        return out


class DenseConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, grate):
        super(DenseConvTranspose2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, grate, (1,3), padding=(0,1))
        self.conv2 = nn.Conv2d(in_channels+grate, grate, (1,3), padding=(0,1))
        self.conv3 = nn.Conv2d(in_channels+2*grate, grate, (1,3), padding=(0,1))
        self.conv4 = nn.Conv2d(in_channels+3*grate, grate, (1,3), padding=(0,1))
        self.conv5 = GluConvTranspose2d(in_channels+4*grate, out_channels, kernel_size, padding=padding, stride=stride)

        self.bn1 = nn.BatchNorm2d(grate)
        self.bn2 = nn.BatchNorm2d(grate)
        self.bn3 = nn.BatchNorm2d(grate)
        self.bn4 = nn.BatchNorm2d(grate)

        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()
        self.elu3 = nn.ELU()
        self.elu4 = nn.ELU()

    def forward(self, x):
        out = x
        out1 = self.elu1(self.bn1(self.conv1(out)))
        out = torch.cat([x, out1], dim=1)
        out2 = self.elu2(self.bn2(self.conv2(out)))
        out = torch.cat([x, out1, out2], dim=1)
        out3 = self.elu3(self.bn3(self.conv3(out)))
        out = torch.cat([x, out1, out2, out3], dim=1)
        out4 = self.elu4(self.bn4(self.conv4(out)))
        out = torch.cat([x, out1, out2, out3, out4], dim=1)
        out5 = self.conv5(out)

        out = out5

        return out

            
class Net(nn.Module):
    def __init__(self, bidirectional):
        super(Net, self).__init__()
                
        self.conv1 = DenseConv2d(4, 16, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv2 = DenseConv2d(16, 32, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv3 = DenseConv2d(32, 64, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv4 = DenseConv2d(64, 128, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv5 = DenseConv2d(128, 256, (1,4), padding=(0,1), stride=(1,2), grate=8)

        self.glstm = GLSTM(5*256, 4, bidirectional)
        
        self.conv5_t = DenseConvTranspose2d(256+256, 256, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv4_t = DenseConvTranspose2d(256+128, 128, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv3_t = DenseConvTranspose2d(128+64, 64, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv2_t = DenseConvTranspose2d(64+32, 32, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv1_t = DenseConvTranspose2d(32+16, 16, (1,4), padding=(0,1), stride=(1,2), grate=8)
        
        self.fc1 = nn.Linear(160*8, 161)
        self.fc2 = nn.Linear(160*8, 161)
        
        self.path1 = DenseConv2d(16, 16, (1,3), padding=(0,1), stride=(1,1), grate=8)
        self.path2 = DenseConv2d(32, 32, (1,3), padding=(0,1), stride=(1,1), grate=8)
        self.path3 = DenseConv2d(64, 64, (1,3), padding=(0,1), stride=(1,1), grate=8)
        self.path4 = DenseConv2d(128, 128, (1,3), padding=(0,1), stride=(1,1), grate=8)
        self.path5 = DenseConv2d(256, 256, (1,3), padding=(0,1), stride=(1,1), grate=8)

    def forward(self, x):
        
        out = x
        e1 = self.conv1(out)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        
        out = e5
        
        out = self.glstm(out)
        
        out = torch.cat([out, self.path5(e5)], dim=1)
        d5 = torch.cat([self.conv5_t(out), self.path4(e4)], dim=1)
        d4 = torch.cat([self.conv4_t(d5), self.path3(e3)], dim=1)
        d3 = torch.cat([self.conv3_t(d4), self.path2(e2)], dim=1)
        d2 = torch.cat([self.conv2_t(d3), self.path1(e1)], dim=1)
        d1 = self.conv1_t(d2)

        out1 = d1[:,:8].transpose(1,2).contiguous().view(d1.size(0), d1.size(2), -1).contiguous()
        out2 = d1[:,8:].transpose(1,2).contiguous().view(d1.size(0), d1.size(2), -1).contiguous()

        out1 = self.fc1(out1)
        out2 = self.fc2(out2)

        out = torch.stack([out1, out2], dim=1)

        return out

class MIMO_Net(nn.Module):
    def __init__(self, bidirectional, net_inp, net_out):
        super(MIMO_Net, self).__init__()
        
        self.net_inp = net_inp
        self.net_out = net_out

        if net_inp==4:
            self.OLD_IMPLEMENTATION = True
        else:
            self.OLD_IMPLEMENTATION = False

        self.conv1 = DenseConv2d(self.net_inp, 16, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv2 = DenseConv2d(16, 32, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv3 = DenseConv2d(32, 64, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv4 = DenseConv2d(64, 128, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv5 = DenseConv2d(128, 256, (1,4), padding=(0,1), stride=(1,2), grate=8)

        self.glstm = GLSTM(5*256, 4, bidirectional)
        
        self.conv5_t = DenseConvTranspose2d(256+256, 256, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv4_t = DenseConvTranspose2d(256+128, 128, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv3_t = DenseConvTranspose2d(128+64, 64, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv2_t = DenseConvTranspose2d(64+32, 32, (1,4), padding=(0,1), stride=(1,2), grate=8)
        self.conv1_t = DenseConvTranspose2d(32+16, 16, (1,4), padding=(0,1), stride=(1,2), grate=8)
        
        if self.OLD_IMPLEMENTATION:
            self.fc1 = nn.Linear(160*8, 161)
            self.fc2 = nn.Linear(160*8, 161)

            if self.net_out==4:
                self.fc3 = nn.Linear(160*8, 161)
                self.fc4 = nn.Linear(160*8, 161)
        else:
            self.linear_layers = nn.ModuleList([ nn.Linear(160*8, 161) for _ in range(0,self.net_out) ])

        self.path1 = DenseConv2d(16, 16, (1,3), padding=(0,1), stride=(1,1), grate=8)
        self.path2 = DenseConv2d(32, 32, (1,3), padding=(0,1), stride=(1,1), grate=8)
        self.path3 = DenseConv2d(64, 64, (1,3), padding=(0,1), stride=(1,1), grate=8)
        self.path4 = DenseConv2d(128, 128, (1,3), padding=(0,1), stride=(1,1), grate=8)
        self.path5 = DenseConv2d(256, 256, (1,3), padding=(0,1), stride=(1,1), grate=8)

    def forward(self, x):
        
        out = x
        e1 = self.conv1(out)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        
        out = e5
        
        out = self.glstm(out)
        
        out = torch.cat([out, self.path5(e5)], dim=1)
        d5 = torch.cat([self.conv5_t(out), self.path4(e4)], dim=1)
        d4 = torch.cat([self.conv4_t(d5), self.path3(e3)], dim=1)
        d3 = torch.cat([self.conv3_t(d4), self.path2(e2)], dim=1)
        d2 = torch.cat([self.conv2_t(d3), self.path1(e1)], dim=1)
        d1 = self.conv1_t(d2)

        out1 = d1[:,:8].transpose(1,2).contiguous().view(d1.size(0), d1.size(2), -1).contiguous()
        out2 = d1[:,8:].transpose(1,2).contiguous().view(d1.size(0), d1.size(2), -1).contiguous()
        #breakpoint()
        if self.OLD_IMPLEMENTATION:
            out_r1 = self.fc1(out1)
            out_i1 = self.fc2(out2)

            out = torch.stack([out_r1, out_i1], dim=1)
            if self.net_out==4:
                out_r2 = self.fc3(out1)
                out_i2 = self.fc4(out2)

                out = torch.stack([out_r1, out_i1, out_r2, out_i2], dim=1)
        else:
            out = torch.stack( [ self.linear_layers[i](out1) if 0==i%2 else self.linear_layers[i](out2) for i in range(0,self.net_out) ], dim=1)

        return out
        
def test_model(bidirectional, num_inp, num_out):
    feat = torch.randn(10, num_inp, 100, 161)
    net = MIMO_Net(bidirectional, num_inp, num_out)
    #print('n_params: {}'.format(numParams(net)))
    #breakpoint()
    #feat = feat.to('cuda:0')
    #net = net.to('cuda:0')
    import time
    start = time.time()
    est = net(feat)
    end = time.time() - start 
    print('Time for {} -> {} is {} sec'.format(feat.shape, est.shape, end))

if __name__=="__main__":
    #import sys
    #bidirectional = sys.argv[1] == "True"
    test_model(bidirectional=True, num_inp=16, num_out=16)