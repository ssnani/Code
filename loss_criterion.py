import torch

class LossFunction(object):
    def __init__(self):
        self.eps = torch.finfo(torch.float32).eps

    def __call__(self, est, lbl):
        batch_size, _, n_frames, n_feats = est.shape

        est_mag = torch.sqrt(est[:,0]**2 + est[:,1]**2 + self.eps)
        lbl_mag = torch.sqrt(lbl[:,0]**2 + lbl[:,1]**2 + self.eps)

        ri = torch.abs(est - lbl)
        mag = torch.abs(est_mag - lbl_mag)

        loss_ri = torch.sum(ri) / float(batch_size* n_frames * n_feats)
        loss_mag = torch.sum(mag) / float(batch_size* n_frames * n_feats)
        loss = loss_ri + loss_mag

        #frame level metrics
        #loss_ri_frm = torch.sum(ri,dim=3)
        #loss_mag_frm = torch.sum(mag,dim=3)

        return loss, loss_ri, loss_mag


if __name__=="__main__":
    loss = LossFunction()
    x = torch.randn(6,2,100,160)
    y = torch.randn(6,2,100,160)
    _loss = loss(x,y)
    print(_loss)
