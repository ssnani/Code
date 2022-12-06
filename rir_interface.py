import scipy.io as sio
import numpy as np

#keys : [0,360) are angle in degree
"""
rirs are created for 
-   Linear 2-Mic array with 10cm
-   Room_size :  [8,8,3]
-   Array_pos :  Room Center
-   Src w.r.t Mic  :  On Circle with radius 1m

"""

class taslp_RIR_Interface():
    #(n_t60,n_angles,n_mics,rir_len)
    def __init__(self):
        self.t60_list = [round(idx,1) for idx in np.arange(0.0,1.1,0.1) if idx!=0.1]
        self.files_list = [f'HABET_SpacedOmni_8x8x3_height1.5_dist1_roomT60_{t60}.mat' for t60 in self.t60_list]
        self.scratch_dir = '/scratch/bbje/battula12/Databases/RIRs/taslp_roomdata_360_resolution_1degree/'
        self.rirs_list, self.dp_rirs_list = self.load_all_rirs()

    
    def load_all_rirs(self):
        lst = []
        dp_lst = []
        for file_name in self.files_list:
            rir = sio.loadmat(f'{self.scratch_dir}{file_name}')
            _h = rir['trainingroom']
            x = np.array(_h.tolist()).squeeze()
            x = np.transpose(x,(0,2,1))
            x = x.astype('float32')
            lst.append(x)
            dp_lst.append(self.get_direct_path_rir(x))
        
        return lst, dp_lst # list of  arrays with shape (360, 2(n_mics), rir_len)

    
    def get_direct_path_rir(self, h):
        #h : (360, 2, rir_len)
        fs = 16000
        correction = int(0.0025*fs)
        h_dp = np.array(h)

        (num_doa, num_ch, rir_len) = h.shape

        idx = np.argmax(np.power(h,2), axis=2)

        for doa_idx in range(num_doa):
            for ch in range(num_ch):
                h_dp[doa_idx, ch, idx[doa_idx, ch]+correction:] = 0

        return h_dp


    def get_rirs(self, t60: float, idx_list: "list integer degrees" ):
        t60_key = self.t60_list.index(t60)
        return self.rirs_list[t60_key][idx_list,:,:], self.dp_rirs_list[t60_key][idx_list,:,:] #(nb_points,  2(n_mics), rir_len))


if __name__=="__main__":
    rir_interface = taslp_RIR_Interface()
    rirs, dp_rirs = rir_interface.get_rirs(t60=0.2, idx_list=[4])
    breakpoint()
    print(rirs.shape)
