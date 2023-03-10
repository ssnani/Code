import torch

class LossFunction(object):
    def __init__(self, loss_flag):
        self.eps = torch.finfo(torch.float32).eps
        self.loss_flag = loss_flag
    def __call__(self, est, lbl):
        batch_size, n_ch, n_frames, n_feats = est.shape

        est_mag_1 = torch.sqrt(est[:,0]**2 + est[:,1]**2 + self.eps)
        lbl_mag_1 = torch.sqrt(lbl[:,0]**2 + lbl[:,1]**2 + self.eps)

        if n_ch==4:
            est_mag_2 = torch.sqrt(est[:,2]**2 + est[:,3]**2 + self.eps)
            lbl_mag_2 = torch.sqrt(lbl[:,2]**2 + lbl[:,3]**2 + self.eps)

            est_mag = est_mag_1 + est_mag_2
            lbl_mag = lbl_mag_1 + lbl_mag_2
        else:
            est_mag = est_mag_1
            lbl_mag = lbl_mag_1

        ri = torch.abs(est - lbl)
        mag = torch.abs(est_mag - lbl_mag)

        loss_ri = torch.sum(ri) / float(batch_size* n_frames * n_feats)
        loss_mag = torch.sum(mag) / float(batch_size* n_frames * n_feats)
        #loss = loss_ri + loss_mag

        #frame level metrics
        #loss_ri_frm = torch.sum(ri,dim=3)
        #loss_mag_frm = torch.sum(mag,dim=3)


        #tried loss_ph over loss_ri -> didn't work
        loss_ph = 0

        if self.loss_flag=="MISO_RI":
            loss = loss_ri
        elif self.loss_flag=="MISO_RI_MAG":
            loss = loss_ri + loss_mag
        else:
            print("Invalid loss flag")

        return loss, loss_ri, loss_mag, loss_ph


class MIMO_LossFunction_v1(object):
    def __init__(self, loss_flag):
        self.eps = torch.finfo(torch.float32).eps
        self.loss_flag = loss_flag

    def __call__(self, est, lbl, epoch):
        batch_size, n_ch, n_frames, n_feats = est.shape

        lbl_mag_1, lbl_ph_1, lbl_mag_2, lbl_ph_2 = self.get_mag_ph(lbl)
        est_mag_1, est_ph_1, est_mag_2, est_ph_2 = self.get_mag_ph(est)

        #print(est_ph_1[0,:,:], est_ph_2[0,:,:])
        ri = torch.abs(est - lbl)
        mag = torch.abs(est_mag_1 - lbl_mag_1) + torch.abs(est_mag_2 - lbl_mag_2)
        

        loss_ri = torch.sum(ri) / float(batch_size* n_frames * n_feats)
        loss_mag = torch.sum(mag) / float(batch_size* n_frames * n_feats)
        
        #loss = loss_ri + loss_mag 

        #ph_diff = torch.abs(torch.cos(lbl_ph_1-lbl_ph_2) - torch.cos(est_ph_1-est_ph_2))
        ph_diff = lbl_mag_1[:,:,1:n_feats-1]*lbl_mag_2[:,:,1:n_feats-1]*torch.abs((lbl_ph_1-lbl_ph_2) - (est_ph_1-est_ph_2))  #best
        #ph_diff = lbl_mag_1[:,:,1:n_feats-1]*lbl_mag_2[:,:,1:n_feats-1]*torch.cos((lbl_ph_1-lbl_ph_2) - (est_ph_1-est_ph_2))   
        #loss_ph_diff = torch.sum(ph_diff) / float(batch_size* n_frames * n_feats)
        #loss -= 0.01*loss_ph_diff #

        ## second order interaction 
        ## R1*R2 + I1*I2, I1R2-R1I2
        #est_r = est[:,0,:,:]*est[:,2,:,:] + est[:,1,:,:]*est[:,3,:,:]
        #lbl_r = lbl[:,0,:,:]*lbl[:,2,:,:] + lbl[:,1,:,:]*lbl[:,3,:,:]

        #est_i = est[:,1,:,:]*est[:,2,:,:] - est[:,0,:,:]*est[:,3,:,:]
        #lbl_i = lbl[:,1,:,:]*lbl[:,2,:,:] - lbl[:,0,:,:]*lbl[:,3,:,:]
        
        #ph_diff = torch.abs(lbl_r-est_r) + torch.abs(lbl_i-est_i)
        loss_ph_diff = torch.sum(ph_diff) / float(batch_size* n_frames * n_feats)
        #loss += 0.01*loss_ph_diff

        #playing with loss 
        #loss = loss_ri + 0.01*loss_ph_diff
        if "MIMO_RI"==self.loss_flag:
            loss = loss_ri
        elif "MIMO_RI_MAG"==self.loss_flag:
            loss = loss_ri + loss_mag
        elif "MIMO_RI_MAG_PD"==self.loss_flag:
            loss = loss_ri + loss_mag + 0.01*loss_ph_diff
        elif "MIMO_RI_PD"==self.loss_flag:
            loss = loss_ri + 0.01*loss_ph_diff
        else:
            print("Invalid loss flag")
        
        #frame level metrics
        #loss_ri_frm = torch.sum(ri,dim=3)
        #loss_mag_frm = torch.sum(mag,dim=3)
        
        # tried ILD didn't work out over IPD , code got deleted
        loss_mag_diff=0

        return loss, loss_ri, loss_mag, loss_ph_diff, loss_mag_diff

    
    def get_mag_ph(self,x):
        #est.shape : (batch, n_ch, T, F)
        (batch, n_ch, T, F) = x.shape
        est_mag_1 = torch.sqrt(x[:,0]**2 + x[:,1]**2 + self.eps)
        est_mag_2 = torch.sqrt(x[:,2]**2 + x[:,3]**2 + self.eps)

        est_ph_1 = torch.atan2(x[:,1,:,1:F-1] + self.eps, x[:,0,:,1:F-1] + self.eps)
        est_ph_2 = torch.atan2(x[:,3,:,1:F-1] + self.eps, x[:,2,:,1:F-1] + self.eps)
        """
        est_c_1 = torch.complex(est[:,0], est[:,1])
        est_c_2 = torch.complex(est[:,2], est[:,3])
        est_mag_1, est_ph_1 = torch.abs(est_c_1), torch.angle(est_c_1)
        est_mag_2, est_ph_2 = torch.abs(est_c_2), torch.angle(est_c_2)
        """
        return est_mag_1, est_ph_1, est_mag_2, est_ph_2


class MIMO_LossFunction(object):
    def __init__(self, loss_flag):
        self.eps = torch.finfo(torch.float32).eps
        self.loss_flag = loss_flag


    def __call__(self, est, lbl, epoch):

        batch_size, n_ch, n_frames, n_feats = est.shape

        num_mics = n_ch//2

        if "MIMO_RI_PD_REF"==self.loss_flag: 
            num_mic_pairs = (num_mics-1)
            mic_pairs = []
            mic_1 = 0
            for mic_2 in range(mic_1+1, num_mics):
                    mic_pairs.append((mic_1, mic_2))

        else:
            num_mic_pairs = (num_mics*(num_mics-1))//2
            mic_pairs = []
            for mic_1 in range(num_mics):
                for mic_2 in range(mic_1+1,num_mics):
                    mic_pairs.append((mic_1, mic_2))
        
            


        lbl_mag = torch.zeros(batch_size, num_mics, n_frames, n_feats)
        est_mag = torch.zeros(batch_size, num_mics, n_frames, n_feats)

        lbl_ph  = torch.zeros(batch_size, num_mics, n_frames, n_feats-2)
        est_ph  = torch.zeros(batch_size, num_mics, n_frames, n_feats-2)
        ph_diff = torch.zeros(batch_size, num_mic_pairs, n_frames, n_feats-2)

        for idx in range(num_mics):
            lbl_mag[:, idx, :,:], lbl_ph[:, idx, :,:] = self.get_mag_ph(lbl[:,idx*2:idx*2+2,:,:])
            est_mag[:, idx, :,:], est_ph[:, idx, :,:] = self.get_mag_ph(est[:,idx*2:idx*2+2,:,:]) 


        #print(est_ph_1[0,:,:], est_ph_2[0,:,:])
        ri = torch.abs(est - lbl)
        mag = torch.abs(est_mag - lbl_mag) # + torch.abs(est_mag_2 - lbl_mag_2)
        
        loss_ri = torch.sum(ri) / float(batch_size* n_frames * n_feats)
        loss_mag = torch.sum(mag) / float(batch_size* n_frames * n_feats)

        for mic_pair_idx, mic_pair in enumerate(mic_pairs):
            ph_diff[:,mic_pair_idx,:,: ] = lbl_mag[:,mic_pair[0],:,1:n_feats-1]*lbl_mag[:,mic_pair[1],:,1:n_feats-1]*torch.abs( (lbl_ph[:,mic_pair[0],:,:]-lbl_ph[:,mic_pair[1],:,:]) - (est_ph[:,mic_pair[0],:,:]-est_ph[:,mic_pair[1],:,:]) )

        #loss = loss_ri + loss_mag 

        #ph_diff = torch.abs(torch.cos(lbl_ph_1-lbl_ph_2) - torch.cos(est_ph_1-est_ph_2))
        #ph_diff = lbl_mag_1[:,:,1:n_feats-1]*lbl_mag_2[:,:,1:n_feats-1]*torch.abs((lbl_ph_1-lbl_ph_2) - (est_ph_1-est_ph_2))  #best
        #ph_diff = lbl_mag_1[:,:,1:n_feats-1]*lbl_mag_2[:,:,1:n_feats-1]*torch.cos((lbl_ph_1-lbl_ph_2) - (est_ph_1-est_ph_2))   
        #loss_ph_diff = torch.sum(ph_diff) / float(batch_size* n_frames * n_feats)
        #loss -= 0.01*loss_ph_diff #

        ## second order interaction 
        ## R1*R2 + I1*I2, I1R2-R1I2
        #est_r = est[:,0,:,:]*est[:,2,:,:] + est[:,1,:,:]*est[:,3,:,:]
        #lbl_r = lbl[:,0,:,:]*lbl[:,2,:,:] + lbl[:,1,:,:]*lbl[:,3,:,:]

        #est_i = est[:,1,:,:]*est[:,2,:,:] - est[:,0,:,:]*est[:,3,:,:]
        #lbl_i = lbl[:,1,:,:]*lbl[:,2,:,:] - lbl[:,0,:,:]*lbl[:,3,:,:]
        
        #ph_diff = torch.abs(lbl_r-est_r) + torch.abs(lbl_i-est_i)
        loss_ph_diff = torch.sum(ph_diff) / float(batch_size * n_frames * n_feats)
        #loss += 0.01*loss_ph_diff

        #playing with loss 
        #loss = loss_ri + 0.01*loss_ph_diff
        if "MIMO_RI"==self.loss_flag:
            loss = loss_ri
        elif "MIMO_RI_MAG"==self.loss_flag:
            loss = loss_ri + loss_mag
        elif "MIMO_RI_MAG_PD" in self.loss_flag:
            loss = loss_ri + loss_mag + 0.01*loss_ph_diff
        elif "MIMO_RI_PD" in self.loss_flag:
            loss = loss_ri + 0.01*loss_ph_diff
        else:
            print("Invalid loss flag")
        
        #frame level metrics
        #loss_ri_frm = torch.sum(ri,dim=3)
        #loss_mag_frm = torch.sum(mag,dim=3)
        
        # tried ILD didn't work out over IPD , code got deleted
        loss_mag_diff=0

        return loss, loss_ri, loss_mag, loss_ph_diff, loss_mag_diff
    
    def get_mag_ph(self,x):
        #est.shape : (batch, n_ch, T, F)
        (batch, n_ch, T, F) = x.shape
        est_mag = torch.sqrt(x[:,0]**2 + x[:,1]**2 + self.eps)
        est_ph = torch.atan2(x[:,1,:,1:F-1] + self.eps, x[:,0,:,1:F-1] + self.eps)
        return est_mag, est_ph

if __name__=="__main__":
    #loss = LossFunction()
    mimo_loss = MIMO_LossFunction("MIMO_RI_PD")
    x = torch.randn(6,8,100,160)
    y = torch.randn(6,8,100,160)
    #_loss = loss(x,y)
    _loss2 = mimo_loss(x,y,0)
    print(_loss2)
    #print(_loss, _loss2)
