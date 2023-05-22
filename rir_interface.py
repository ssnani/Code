import scipy.io as sio
import numpy as np

#keys : [0,360) are angle in degree
"""
rirs are created for 
-   Linear 2-Mic array with 10cm, 8cm
-   Room_size :  [8,8,3], [6,6,2.4]
-   Array_pos :  Room Center
-   Src w.r.t Mic  :  On Circle with radius 1m
"""

# Add support different mic distance for linear array or different geometry
#
class taslp_RIR_Interface():
    #(n_t60,n_angles,n_mics,rir_len)
    #currently (2, 8 ) channel rirs
    def __init__(self, array_type: str, num_mics:int, intermic_dist: float, room_size: list, train_flag: bool, sub_mics: int = -1):
        self.t60_list = [round(idx,1) for idx in np.arange(0.0,1.1,0.1) if idx!=0.1] 
        self.files_list = [f'HABET_SpacedOmni_{room_size[0]}x{room_size[1]}x{room_size[2]}_height{float(room_size[2])/2}_dist1_roomT60_{t60}.mat' for t60 in self.t60_list]
        self.file_num_mics = 8 if 'linear' in array_type else 7 #if( num_mics > 2 and num_mics <= 8) else 2
        self.scratch_dir = f'/fs/scratch/PAS0774/Shanmukh/Databases/RIRs/taslp_roomdata_360_resolution_1degree_{array_type}_array_{self.file_num_mics}_mic_{intermic_dist}cm/'
        self.rirs_list, self.dp_rirs_list = self.load_all_rirs(train_flag)
        self.req_mics = num_mics #sub_mics if sub_mics > 0 and sub_mics !=num_mics else num_mics

    def load_all_rirs(self, train_flag):
        lst = []
        dp_lst = []
        _str = 'trainingroom' if train_flag else 'testingroom'
        for file_name in self.files_list:
            rir = sio.loadmat(f'{self.scratch_dir}{file_name}')
            _h = rir[_str]  # testingroom
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
        if self.req_mics==self.file_num_mics:
            return self.rirs_list[t60_key][idx_list,:,:], self.rirs_list[0][idx_list,:,:] #self.dp_rirs_list[t60_key][idx_list,:,:] #(nb_points,  2(n_mics), rir_len))
        else:
            #picking centre mics
            #login written for even mics
            idx = self.file_num_mics//2
            mic_pair_idx_offset = self.req_mics//2
            return self.rirs_list[t60_key][idx_list,idx-mic_pair_idx_offset:idx+mic_pair_idx_offset,:], self.rirs_list[0][idx_list,idx-mic_pair_idx_offset:idx+mic_pair_idx_offset,:] #self.dp_rirs_list[t60_key][idx_list,:,:] #(nb_points,  2(n_mics), rir_len))

class taslp_real_RIR_Interface():
    #(n_t60,n_angles,n_mics,rir_len)
    #dist: [1, 2]m
    def __init__(self, dist:int, num_mics):
        self.t60_list = [0.16, 0.36, 0.61]
        self.files_list = [f'aachen_8-8-8-8-8-8-8_roomT60_{t60}.mat' for t60 in self.t60_list] 
        self.scratch_dir = f'/fs/scratch/PAS0774/Shanmukh/Databases/RIRs/taslp_aachen_real_rirs/'
        self.dist = dist
        self.idx_offset = 0 if 1==dist else 13 
        self.rirs_list, self.dp_rirs_list = self.load_all_rirs()
        self.file_num_mics = 8
        self.num_mics = num_mics
    # idx-> degree
    # 0  -> 180 
    def load_all_rirs(self):
        lst = []
        dp_lst = []
        idx_strt, idx_end = 0+self.idx_offset, 13+self.idx_offset # (0-12 (1m), (13-25) (2m))
        for file_name in self.files_list:
            rir = sio.loadmat(f'{self.scratch_dir}{file_name}')
            _h = rir['testingroom']  # testingroom
            x = np.array(_h.tolist()).squeeze()
            x = np.transpose(x,(0,2,1))
            x = x.astype('float32') 
            lst.append(x[idx_strt:idx_end,:,:])
            dp_lst.append(self.get_direct_path_rir(x[idx_strt:idx_end,:,:]))
        
        return lst, dp_lst # list of  arrays with shape (13, 8(n_mics), rir_len) 

    
    def get_direct_path_rir(self, h):
        #h : (26, 8, rir_len)
        fs = 16000
        correction = 1 #int(0.0025*fs)
        h_dp = np.array(h)

        (num_doa, num_ch, rir_len) = h.shape

        idx = np.argmax(np.power(h,2), axis=2)

        for doa_idx in range(num_doa):
            for ch in range(num_ch):
                h_dp[doa_idx, ch, idx[doa_idx, ch]+correction:] = 0

        return h_dp


    def get_rirs(self, t60: float, idx_list: "list integer degrees" ):
        t60_key = self.t60_list.index(t60)
        #rir
        idx_list = [12-idx for idx in idx_list]
        mic_centre_idx = self.file_num_mics//2
        mic_idx_list = self.num_mics//2
        return self.rirs_list[t60_key][idx_list,mic_centre_idx-mic_idx_list:mic_centre_idx+mic_idx_list,:], self.dp_rirs_list[t60_key][idx_list,mic_centre_idx-mic_idx_list:mic_centre_idx+mic_idx_list,:] #(nb_points,  2(n_mics), rir_len)) picking 8cm intermic dist (3:5)

class gannot_sim_RIR_Interface():
    #my simulation only for 8cm
    def __init__(self, dist: int):
        self.t60_list = [0.16, 0.36, 0.61]
        self.files_list = [f'HABET_SpacedOmni_6x6x2.4_height1.2_dist1_roomT60_{t60}s.mat' for t60 in self.t60_list]
        self.scratch_dir = f'/fs/scratch/PAS0774/Shanmukh/Databases/RIRs/'
        self.rirs_list, self.dp_rirs_list = self.load_all_rirs()
   
    def load_all_rirs(self):
        lst = []
        dp_lst = []
        _str = 'testingroom'
        for file_name in self.files_list:
            rir = sio.loadmat(f'{self.scratch_dir}{file_name}')
            _h = rir[_str]  # testingroom
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
        return self.rirs_list[t60_key][idx_list,:,:], self.rirs_list[0][idx_list,:,:] #self.dp_rirs_list[t60_key][idx_list,:,:] #(nb_points,  2(n_mics), rir_len))

if __name__=="__main__":
    #array_type, num_mics, intermic_dist,  room_size = 'linear', 2, 8.0,  ['6', '6', '2.4']
    array_type, num_mics, intermic_dist,  room_size = 'circular', 7, 4.25,  ['6', '6', '2.4']
    
    rir_interface = taslp_RIR_Interface(array_type, num_mics, intermic_dist, room_size, None)
    rirs, dp_rirs = rir_interface.get_rirs(t60=0.2, idx_list=[75])
    #rirs_0, dp_rirs_0 = rir_interface.get_rirs(t60=0.0, idx_list=[4])

    tr_rir_interface = taslp_RIR_Interface(array_type, 4, intermic_dist, room_size, None)
    tr_rirs, tr_dp_rirs = tr_rir_interface.get_rirs(t60=0.2, idx_list=[75])#30, 360-30, 180-30, 180+30])

    #tr_rir_interface_4 = taslp_RIR_Interface(array_type, 4, intermic_dist, room_size, True)
    #tr_rirs_4, tr_dp_rirs_4 = tr_rir_interface_4.get_rirs(t60=0.6, idx_list=[10,360-10, 180-10])
    #rirs_0, dp_rirs_0 = rir_interface.get_rirs(t60=0.0, idx_list=[4])

    breakpoint()
    """
    real_rir_interface = taslp_real_RIR_Interface(dist=1)
    rirs, dp_rirs = real_rir_interface.get_rirs(t60=0.16, idx_list=[4])
    breakpoint()
    """
    print(rirs.shape)
