import torch
import numpy as np

import torchaudio

#GCC-PHAT (2D)

# inp: comp_spec (chan, Time, freq)
#mic_pos: local mic_pos
#mic_center: local mic_center 


# **** Convention: doa w.r.t X-axis  *****

"""
local_mic_pos = torch.from_numpy(np.array((
                    ( 0.05, 0.00, 0.00),
                    ( 0.00, 0.00, 0.00),
                    ( -0.05, 0.00, 0.00)
                    )))

local_mic_center = local_mic_pos[[1],:]
mic_pair = (0,2)
local_mic_pos = torch.concat((local_mic_pos[[mic_pair[0]],:], local_mic_pos[[mic_pair[1]],:]), axis=0)
"""           

#torch implementation
def gcc_phat_loc_orient(X, est_mask, fs, nfft, local_mic_pos, mic_center, src_mic_dist, weighted, sig_vad, is_euclidean_dist):

    (chan, frames, freq) = X.shape
    X_ph = torch.angle(X)
    X_ph_diff = X_ph[0,:,1:] - X_ph[1,:,1:]

    #weightage
    if weighted:
        est_mask_pq = torch.pow(est_mask[0,:,1:]*est_mask[1,:,1:], 1) # 0.3) # 0.5


    angular_freq = 2*torch.pi*fs*1.0/nfft*torch.arange(1, freq, dtype=torch.float32)

    angle_step = 1.0
    theta_grid = 360.0 if mic_center=="circular" else 180.0
    all_directions = torch.linspace(0,theta_grid,int(theta_grid/angle_step+1),dtype=torch.float32) #a set of potential directions
    dist = src_mic_dist
    c = 343

    all_directions_val = [] 
    delays = []
    all_info = []
    all_info_unweighted = []

    for ind, direction in enumerate(all_directions):
        #ang = (direction - 90)/180.0*torch.pi  #radians
        ang_doa = (direction)/180*torch.pi  #radians
        
        if is_euclidean_dist:
            src_pos = torch.tensor([torch.cos(ang_doa)*dist,torch.sin(ang_doa)*dist, 0.0],dtype=torch.float32) #+ mic_center
            dist_pp = torch.sqrt(torch.sum((src_pos-local_mic_pos[0])**2))  ## TODO: pp
            dist_qq = torch.sqrt(torch.sum((src_pos-local_mic_pos[1])**2))  ## TODO: qq
            delay = (dist_qq-dist_pp)/c #,device=est_mask.device)#.type_as(est_mask)   

        else:
            # ASSUMPTION on unit circle
            dist = 1
            src_pos = torch.tensor([torch.cos(ang_doa)*dist,torch.sin(ang_doa)*dist, 0.0],dtype=torch.float32) 
            delay = np.dot((local_mic_pos[0]-local_mic_pos[1]), src_pos)/c

            
        delays.append(delay)
        delay_vec = angular_freq*delay
        #print(X_ph_diff.shape, delay_vec.shape)
        gcc_phat_pq = torch.cos( X_ph_diff - delay_vec.to(device=X_ph_diff.device))

        all_info_unweighted.append(gcc_phat_pq)   #debug
        if weighted:
            mgcc_phat_pq = est_mask_pq*gcc_phat_pq
            gcc_phat_pq = mgcc_phat_pq

        all_info.append(gcc_phat_pq)              #debug
        per_direction_val = torch.sum(gcc_phat_pq, dim=1)

        all_directions_val.append(per_direction_val)

    vals = torch.stack(all_directions_val, dim=0)

    sig_vad_frms = sig_vad.shape[0]
    vals = vals[:,:sig_vad_frms]*sig_vad   ##caluation for only vad frames

    utt_sum = torch.sum(vals,dim=1)
    utt_doa_idx = torch.argmax(utt_sum)
    utt_doa = all_directions[utt_doa_idx]


    doa_idx = torch.argmax(vals,dim=0)
    doa = all_directions[doa_idx.to(all_directions.device)]

    return doa, vals, utt_doa, all_info_unweighted, all_info #utt_sum, delays

def gcc_phat_all_pairs(X: "[num_mics, T, F]", est_mask, fs, nfft, local_mic_pos, mic_center, src_mic_dist, weighted, sig_vad, is_euclidean_dist, mic_pairs):
    #print(X.shape)
    num_mics, num_frames, num_freq = X.shape
    #mic_pairs = [(mic_1, mic_2) for mic_1 in range(0, num_mics) for mic_2 in range(mic_1+1, num_mics)]
    theta_grid, mic_center = (361, "circular") if num_mics==7 else (181, "linear")
    pair_acc_X_frm_vals = torch.zeros(theta_grid, num_frames).to(X.device)  #theta_grid [0,180]

    #centre mics (2mic)
    #  Even mics
    idx = num_mics//2
    centre_mic_pair = (idx-1, idx) if num_mics!=7 else (1,4)  #'linear' in array_type 
    #print(centre_mic_pair)
   
    for mic_pair in mic_pairs:
        X_pair = X[[mic_pair[0], mic_pair[1]],:,:]
        est_mask_pair = est_mask[[mic_pair[0], mic_pair[1]],:,:]
        mic_pair_pos = local_mic_pos[[mic_pair[0], mic_pair[1]],:]

        X_pair_doa, X_frm_vals, X_pair_utt_doa, _, _ = gcc_phat_loc_orient(X_pair, est_mask_pair, fs, nfft, mic_pair_pos, 
															 mic_center, src_mic_dist, weighted, sig_vad, is_euclidean_dist)
        
        if centre_mic_pair == mic_pair:
            X_2mic_doa = X_pair_doa
            X_2mic_utt_doa = X_pair_utt_doa
            X_2mic_frm_vals = X_frm_vals

        pair_acc_X_frm_vals += X_frm_vals

    doa_idx = torch.argmax(pair_acc_X_frm_vals,dim=0)  #assuming doa_idx is doa

    utt_sum = torch.sum(pair_acc_X_frm_vals,dim=1)
    utt_doa_idx = torch.argmax(utt_sum)

    return doa_idx, pair_acc_X_frm_vals, utt_doa_idx, X_2mic_doa, X_2mic_utt_doa, X_2mic_frm_vals

def np_gcc_phat_loc_orient(X, est_mask, fs, nfft, local_mic_pos, mic_center, src_mic_dist, weighted, sig_vad, is_euclidean_dist):

    (chan, frames, freq) = X.shape
    X_ph = np.angle(X)
    X_ph_diff = X_ph[0,:,1:] - X_ph[1,:,1:]

    #weightage
    if weighted:
        est_mask_pq = np.power(est_mask[0,:,1:]*est_mask[1,:,1:], 1.0) # 0.3) # 


    angular_freq = 2*np.pi*fs*1.0/nfft*np.arange(1, freq, dtype=np.float32)

    angle_step = 1.0
    theta_grid = 360.0 if mic_center=="circular" else 180.0
    all_directions = np.linspace(0,theta_grid,int(theta_grid/angle_step+1),dtype=np.float32) #a set of potential directions
    dist = src_mic_dist
    c = 343

    all_directions_val = [] 
    delays = []
    all_info = []
    all_info_unweighted = []

    for ind, direction in enumerate(all_directions):
        #ang = (direction - 90)/180.0*torch.pi  #radians
        ang_doa  = (direction)/180*np.pi  #radians
        
        if is_euclidean_dist:
            src_pos = np.array([np.cos(ang_doa)*dist,np.sin(ang_doa)*dist, 0.0],dtype=np.float32) #+ mic_center
            dist_pp = np.sqrt(np.sum((src_pos-local_mic_pos[0])**2))  ## TODO: pp
            dist_qq = np.sqrt(np.sum((src_pos-local_mic_pos[1])**2))  ## TODO: qq
            delay = (dist_qq-dist_pp)/c #,device=est_mask.device)#.type_as(est_mask)   

        else:
            # ASSUMPTION on unit circle
            dist = 1
            src_pos = np.array([np.cos(ang_doa)*dist,np.sin(ang_doa)*dist, 0.0],dtype=np.float32)
            delay = np.dot((local_mic_pos[0]-local_mic_pos[1]), src_pos)/c

            
        delays.append(delay)
        delay_vec = angular_freq*delay
        #print(X_ph_diff.shape, delay_vec.shape)
        gcc_phat_pq = np.cos( X_ph_diff - delay_vec)#.to(device=X_ph_diff.device)

        all_info_unweighted.append(gcc_phat_pq)   #debug
        if weighted:
            mgcc_phat_pq = est_mask_pq*gcc_phat_pq
            gcc_phat_pq = mgcc_phat_pq

        all_info.append(gcc_phat_pq)              #debug
        per_direction_val = np.sum(gcc_phat_pq, axis=1)

        all_directions_val.append(per_direction_val)

    vals = np.stack(all_directions_val, axis=0)

    sig_vad_frms = sig_vad.shape[0]
    vals = vals[:,:sig_vad_frms]*sig_vad   ##caluation for only vad frames

    utt_sum = np.sum(vals,axis=1)
    utt_doa_idx = np.argmax(utt_sum)
    utt_doa = all_directions[utt_doa_idx]

    doa_idx = np.argmax(vals,axis=0)
    doa = all_directions[doa_idx]#.to(all_directions.device)]

    return doa, vals, utt_doa, all_info_unweighted, all_info #utt_sum, delays

def np_gcc_phat_all_pairs(X: "[num_mics, T, F]", est_mask, fs, nfft, local_mic_pos, mic_center, src_mic_dist, weighted, sig_vad, is_euclidean_dist, mic_pairs):
    #print(X.shape)
    num_mics, num_frames, num_freq = X.shape
    #mic_pairs = [(mic_1, mic_2) for mic_1 in range(0, num_mics) for mic_2 in range(mic_1+1, num_mics)]
    theta_grid, mic_center = (361, "circular") if num_mics==7 else (181, "linear")
    pair_acc_X_frm_vals = np.zeros((theta_grid, num_frames)) #.to(X.device)  #theta_grid [0,180]

    #centre mics (2mic)
    #  Even mics
    idx = num_mics//2
    centre_mic_pair = (idx-1, idx) if num_mics!=7 else (1,4)  #'linear' in array_type 
    #print(centre_mic_pair)
   
    for mic_pair in mic_pairs:
        X_pair = X[[mic_pair[0], mic_pair[1]],:,:]
        est_mask_pair = est_mask[[mic_pair[0], mic_pair[1]],:,:]
        mic_pair_pos = local_mic_pos[[mic_pair[0], mic_pair[1]],:]

        X_pair_doa, X_frm_vals, X_pair_utt_doa, _, _ = np_gcc_phat_loc_orient(X_pair, est_mask_pair, fs, nfft, mic_pair_pos, 
															 mic_center, src_mic_dist, weighted, sig_vad, is_euclidean_dist)
        
        if centre_mic_pair == mic_pair:
            X_2mic_doa = X_pair_doa
            X_2mic_utt_doa = X_pair_utt_doa
            X_2mic_frm_vals = X_frm_vals

        pair_acc_X_frm_vals += X_frm_vals

    doa_idx = np.argmax(pair_acc_X_frm_vals,axis=0)  #assuming doa_idx is doa

    utt_sum = np.sum(pair_acc_X_frm_vals,axis=1)
    utt_doa_idx = np.argmax(utt_sum)

    return doa_idx, pair_acc_X_frm_vals, utt_doa_idx, X_2mic_doa, X_2mic_utt_doa, X_2mic_frm_vals
    
#Vad 
#input: numpy signal
#output: torch signal
def compute_vad(source_signal: 'Array ( sig_len)', frame_size: 'int (samples) ', frame_shift: 'int (samples)', fs: int = 16000):
    import webrtcvad
    vad = webrtcvad.Vad()
    agressiveness=3
    vad.set_mode(agressiveness)

    #sig_vad = np.zeros_like(source_signal)
    sig_vad = []
    sig_size = source_signal.shape[-1]

    for frame_idx, frm_strt in enumerate(range(0, sig_size-frame_size +1, frame_shift)):
        frame = source_signal[frm_strt:frm_strt + frame_size] #source_signal[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
        frame_bytes = (frame * 32767).astype('int16').tobytes()
        #sig_vad[frm_strt:frm_strt + frame_size] = vad.is_speech(frame_bytes, fs)
        sig_vad.append(vad.is_speech(frame_bytes, fs))

    return torch.tensor(np.array(sig_vad))

#wrapper
# input: td (2, *) 
def gcc_phat(sig: "shape: (2,*)", local_mic_pos, local_mic_center, src_mic_dist, weighted, is_euclidean_dist):
    nfft=320
    fs, c, freq_bins = 16000, 343, np.arange(1, nfft//2) 

    if 0:
        #Vad 
        sig_vad_1 = compute_vad(sig[0,:].numpy(), nfft, nfft//2)
        sig_vad_2 = compute_vad(sig[1,:].numpy(), nfft, nfft//2)
        sig_vad = sig_vad_1*sig_vad_2
    else:
        sig_vad = compute_vad_speech_brain(sig, nfft , nfft//2)

    X = torch.stft(sig, nfft, nfft // 2, nfft, torch.hamming_window(nfft), return_complex=True, center = False)
    #X = X[:,1:,:] # removing the dc component
    X = torch.permute(X, [0,2,1])

    f_doa, f_vals, utt_doa, utt_sum, delays = gcc_phat_loc_orient(X, torch.abs(X), fs, nfft, local_mic_pos, local_mic_center, src_mic_dist, weighted, sig_vad, is_euclidean_dist)
    
    return utt_doa, utt_sum, f_doa, f_vals, sig_vad, delays



def gcc_phat_v2(sig: "shape: (2,*)", local_mic_pos, local_mic_center, src_mic_dist, weighted, is_euclidean_dist, sig_vad):
    nfft=320
    fs, c, freq_bins = 16000, 343, np.arange(1, nfft//2) 

    X = torch.stft(sig, nfft, nfft // 2, nfft, torch.hamming_window(nfft), return_complex=True, center = False)
    #X = X[:,1:,:] # removing the dc component
    X = torch.permute(X, [0,2,1])

    f_doa, f_vals, utt_doa, utt_sum, delays = gcc_phat_loc_orient(X, torch.abs(X), fs, nfft, local_mic_pos, local_mic_center, src_mic_dist, weighted, sig_vad, is_euclidean_dist)
    
    return utt_doa, utt_sum, f_doa, f_vals, sig_vad, delays
    

def compute_vad_speech_brain(sig, frame_size: 'int (samples) ', frame_shift: 'int (samples)', fs: int = 16000):
    from speechbrain.pretrained import VAD
    VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")

    #sig, fs = torchaudio.load(file_path)
    
    torchaudio.save("ch_0.wav", sig[[0]], sample_rate=16000)
    torchaudio.save("ch_1.wav", sig[[1]], sample_rate=16000)

    boundaries_0 = VAD.get_speech_segments("ch_0.wav")
    boundaries_1 = VAD.get_speech_segments("ch_1.wav")

    upsampled_boundaries_0 = VAD.upsample_boundaries(boundaries_0, "ch_0.wav")
    upsampled_boundaries_1 = VAD.upsample_boundaries(boundaries_1, "ch_1.wav")

    boundary = upsampled_boundaries_0*upsampled_boundaries_1

    sig_vad = []
    sig_size = sig.shape[-1]
    for frame_idx, frm_strt in enumerate(range(0, sig_size-frame_size +1, frame_shift)):
        frame_vad = torch.mean(boundary[0, frm_strt:frm_strt + frame_size])
        frame_dec = 1 if frame_vad > 0.95 else 0
        sig_vad.append(frame_dec)

    return torch.tensor(np.array(sig_vad))#, boundary

    



def block_doa(frm_val, block_size):
    n_blocks = frm_val.shape[1]// block_size + 1

    blk_doas = []
    blk_start = 0
    for blk_idx in range(0, n_blocks):

        if blk_idx < n_blocks-1:
            blk_end = blk_start + block_size
        else:
            blk_end = blk_start + frm_val.shape[1]

        blk_sum = torch.sum(frm_val[:,blk_start:blk_end],dim=1)
        blk_doa_idx = torch.argmax(blk_sum)
        blk_doa = blk_doa_idx

        blk_start = blk_end

        blk_doas.append(blk_doa)

    return blk_doas

#for now taking absolute inside the block
def block_lbl_doa(lbl_doa, block_size):
    #assuming lbl_doa [r, elv, azi] : (1, frms, 3)
    n_blocks = lbl_doa.shape[1]// block_size + 1

    blk_doas = []
    blk_range = []
    blk_start = 0
    for blk_idx in range(0, n_blocks):
        if blk_idx < n_blocks-1:
            blk_end = blk_start + block_size
        else:
            blk_end = blk_start + lbl_doa.shape[1]

        blk_doa = torch.mean(torch.abs(lbl_doa[:,blk_start:blk_end, : ]),dim=1)

        blk_start = blk_end
        blk_doas.append(torch.rad2deg(blk_doa[:,2])) # azimuth
        blk_range.append(blk_doa[:,0]) # range

    return blk_doas, blk_range


def blk_vad(frm_level_vad, block_size):
    #assuming frm_level_vad : (frms)
    n_blocks = frm_level_vad.shape[0]// block_size + 1

    frm_level_vad = 1.0*frm_level_vad
    blk_vads = []
    blk_start = 0
    for blk_idx in range(0, n_blocks):
        if blk_idx < n_blocks-1:
            blk_end = blk_start + block_size
        else:
            blk_end = blk_start + frm_level_vad.shape[0]

        _blk_vad = torch.mean(frm_level_vad[blk_start:blk_end])

        blk_start = blk_end
        blk_vads.append(_blk_vad)
    return blk_vads



#torch implementation : yet to be implemented
def batch_gcc_phat_loc_orient(X, est_mask, fs, nfft, local_mic_pos, mic_center, src_mic_dist, weighted, sig_vad, is_euclidean_dist):

    (chan, frames, freq) = X.shape
    X_ph = torch.angle(X)
    X_ph_diff = X_ph[0,:,:] - X_ph[1,:,:]

    #weightage
    if weighted:
        est_mask_pq = torch.power((est_mask[0,:,:]*est_mask[1,:,:]), 0.3)


    angular_freq = 2*torch.pi*fs*1.0/nfft*torch.arange(0, freq, dtype=torch.float32)

    angle_step = 1.0
    all_directions = torch.linspace(0,180.0,int(2*90.0/angle_step+1),dtype=torch.float32) #a set of potential directions
    dist = src_mic_dist
    c = 343

    all_directions_val = [] 
    delays = []

    for ind, direction in enumerate(all_directions):
        #ang = (direction - 90)/180.0*torch.pi  #radians
        ang_doa  = (direction)/180.0*torch.pi  #radians
        
        if is_euclidean_dist:
            src_pos = torch.tensor([torch.cos(ang_doa)*dist,torch.sin(ang_doa)*dist, 0.0],dtype=torch.float32) #+ mic_center
            dist_pp = torch.sqrt(torch.sum((src_pos-local_mic_pos[0])**2))  ## TODO: pp
            dist_qq = torch.sqrt(torch.sum((src_pos-local_mic_pos[1])**2))  ## TODO: qq
            delay = (dist_qq-dist_pp)/c #,device=est_mask.device)#.type_as(est_mask)   

        else:
            # ASSUMPTION on unit circle
            dist = 1
            src_pos = torch.tensor([torch.cos(ang_doa)*dist,torch.sin(ang_doa)*dist, 0.0],dtype=torch.float32) 
            delay = np.dot((local_mic_pos[0]-local_mic_pos[1]), src_pos)/c

            
        delays.append(delay)
        delay_vec = angular_freq*delay
        #print(X_ph_diff.shape, delay_vec.shape)
        gcc_phat_pq = torch.cos( X_ph_diff - delay_vec.to(device=X_ph_diff.device))
        if weighted:
            mgcc_phat_pq = est_mask_pq*gcc_phat_pq
            gcc_phat_pq = mgcc_phat_pq

        per_direction_val = torch.sum(gcc_phat_pq, dim=1)

        all_directions_val.append(per_direction_val)

    vals = torch.stack(all_directions_val, dim=0)

    sig_vad_frms = sig_vad.shape[0]
    vals = vals[:,:sig_vad_frms]*sig_vad   ##caluation for only vad frames

    utt_sum = torch.sum(vals,dim=1)
    utt_doa_idx = torch.argmax(utt_sum)
    utt_doa = all_directions[utt_doa_idx]


    doa_idx = torch.argmax(vals,dim=0)
    doa = all_directions[doa_idx]

    return doa, vals, utt_doa, utt_sum, delays
