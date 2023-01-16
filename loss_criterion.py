import torch

class LossFunction(object):
    def __init__(self):
        self.eps = torch.finfo(torch.float32).eps

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
        loss = loss_ri + loss_mag

        #frame level metrics
        #loss_ri_frm = torch.sum(ri,dim=3)
        #loss_mag_frm = torch.sum(mag,dim=3)


        return loss, loss_ri, loss_mag



class MIMO_LossFunction(object):
    def __init__(self):
        self.eps = torch.finfo(torch.float32).eps

    def __call__(self, est, lbl):
        batch_size, n_ch, n_frames, n_feats = est.shape

        est_mag_1, est_ph_1, est_mag_2, est_ph_2 = self.get_mag_ph(est)
        lbl_mag_1, lbl_ph_1, lbl_mag_2, lbl_ph_2 = self.get_mag_ph(lbl)

        ri = torch.abs(est - lbl)
        mag = torch.abs(est_mag_1 - lbl_mag_1) + torch.abs(est_mag_2 - lbl_mag_2)
        #ph = torch.abs((lbl_ph_1-lbl_ph_2) - (est_ph_1-est_ph_2))

        loss_ri = torch.sum(ri) / float(batch_size* n_frames * n_feats)
        loss_mag = torch.sum(mag) / float(batch_size* n_frames * n_feats)
        loss = loss_ri + loss_mag

        #frame level metrics
        #loss_ri_frm = torch.sum(ri,dim=3)
        #loss_mag_frm = torch.sum(mag,dim=3)

        return loss, loss_ri, loss_mag

    
    def get_mag_ph(self,est):
        #est.shape : (batch, n_ch, T, F)
        est_c_1 = torch.complex(est[:,0], est[:,1])
        est_c_2 = torch.complex(est[:,2], est[:,3])
        est_mag_1, est_ph_1 = torch.abs(est_c_1), torch.angle(est_c_1)
        est_mag_2, est_ph_2 = torch.abs(est_c_2), torch.angle(est_c_2)

        return est_mag_1, est_ph_1, est_mag_2, est_ph_2



if __name__=="__main__":
    loss = LossFunction()
    mimo_loss = MIMO_LossFunction()
    x = torch.randn(6,4,100,160)
    y = torch.randn(6,4,100,160)
    _loss = loss(x,y)
    _loss2 = mimo_loss(x,y)
    print(_loss, _loss2)
