import numpy as np
import os
import webrtcvad
import soundfile

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

import gpuRIR
import random
from collections import namedtuple
from utils import Parameter
from locata_utils import *

from debug import dbg_print

from controlled_config import ControlledConfig
from rir_interface import taslp_RIR_Interface
from array_setup import array_setup_10cm_2mic
from locata_utils import cart2sph

class NetworkInput(object):
	def __init__(self, frame_len, frame_shift, ref_mic_idx=0):
		self.frame_len = frame_len
		self.frame_shift = frame_shift
		self.kernel = (1,frame_len)
		self.stride = (1,frame_shift)
		self.mic_idx = ref_mic_idx   

	def __call__(self, mix, tgt, DOA):
		# Expecting shape mix, tgt: (2, *)
		# acoustic_scene has numpy objects doa (samples, 2)
		# Output shape: (2*2(r,i), T, f), (T,2)
		dbg_print(f"input : mix: {mix.shape} tgt: {tgt.shape},  doa:{DOA.shape}")
		ip_real_img = torch.stft(mix, self.frame_len, self.frame_shift, self.frame_len, torch.hamming_window(self.frame_len), center=False, return_complex=False) #(2, num_freq, T, 2(r,i))
		tgt_real_img = torch.stft(tgt, self.frame_len, self.frame_shift, self.frame_len, torch.hamming_window(self.frame_len), center=False, return_complex=False)

		ip_real_img = torch.permute(ip_real_img, [0, 3, 2, 1]) #(2, 2(r,i), T, F)
		tgt_real_img = torch.permute(tgt_real_img, [0, 3, 2, 1]) #(2, 2(r,i), T, F)

		(num_ch, _ri, frms, freq) = ip_real_img.shape
		ip_feat = torch.reshape(ip_real_img, (num_ch*_ri, frms, freq))
		tgt_feat = torch.reshape(tgt_real_img, (num_ch*_ri, frms, freq))


		# Code for frame_level doa
		doa = torch.from_numpy(DOA.T)
		doa = doa.unsqueeze(dim=1).unsqueeze(dim=1) #reshape required for unfold
		doa_frm = torch.nn.functional.unfold(doa, self.kernel, padding=(0,0), stride=self.stride)
		doa_frm = torch.mode(doa_frm, dim=1).values.T  

		#float32 for pytorch lightning
		ip_feat, tgt_feat, doa_frm = ip_feat.to(torch.float32), tgt_feat.to(torch.float32), doa_frm.to(torch.float32)

		#MISO 
		if -1 != self.mic_idx:       #condition for specific channels
			tgt_feat = tgt_feat[2*self.mic_idx :2*self.mic_idx  + 2]

		dbg_print(f"Transform inp: {ip_feat.shape} tgt: {tgt_feat.shape},  doa:{doa_frm.shape}")
		return ip_feat, tgt_feat, doa_frm  


class MovingSourceDataset(Dataset):

	# Currently supported for > 10 sec utterances

	def __init__(self, dataset_info_file, array_setup, transforms: list =None, size=None):
		
		with open(dataset_info_file, 'r') as f:
			self.tr_ex_list = [line.strip() for line in f.readlines()]

		self.fs = 16000
		self.resample = torchaudio.transforms.Resample(new_freq = self.fs)
		self.array_setup = array_setup
		self.transforms = transforms if transforms is not None else None
		self.size = size

		self.rir_interface = taslp_RIR_Interface() 

	def __len__(self):
		return len(self.tr_ex_list) if self.size is None else self.size

	def __getitem__(self, idx):
		line_info = self.tr_ex_list[idx].split(',')

		#sph
		sph, fs = torchaudio.load(line_info[0])
		noi, fs_noi = torchaudio.load(line_info[1])
		cfg = torch.load(line_info[2])

		dbg_print(cfg)
		if self.fs != fs_noi:
			noi = torch.resample(noi)

		if self.fs != fs:
			sph = torch.resample(sph)

		#truncating to 10 sec utterances
		

		sph_len = sph.shape[1]
		noi_len = noi.shape[1]

		dbg_print(f'sph: {sph_len}, noi: {noi_len}')
		assert sph_len > 16000*4       #TODO: tempchange to 4 sec
		sph_len = 16000*4
		sph = sph[:,:sph_len]

		if sph_len < noi_len:
			"""
			start_idx = random.randint(0, noi_len - sph_len -1)
			noi = noi[:,start_idx:start_idx+sph_len]
			"""
			noi = noi[:,:sph_len]
		else:
			sph = sph[:,:noi_len]

		dbg_print(f'sph: {sph_len}, noi: {noi_len}')

		mic_pos = cfg['mic_pos']
		array_pos = cfg['array_pos'] #np.mean(mic_pos, axis=0)
		#beta = gpuRIR.beta_SabineEstimation(cfg['room_sz'], cfg['t60'], cfg['abs_weights'])

		"""
		self.nb_points = cfg['src_traj_pts'].shape[0]
		traj_pts = cfg['src_traj_pts']
		
		# Interpolate trajectory points
		timestamps = np.arange(self.nb_points) * sph_len / self.fs / self.nb_points
		t = np.arange(sph_len)/self.fs
		trajectory = np.array([np.interp(t, timestamps, traj_pts[:,i]) for i in range(3)]).transpose()

		noise_pos = cfg['noise_pos']
		noise_pos = np.expand_dims(noise_pos, axis=0) if noise_pos.shape[0] != 1 else noise_pos

		noise_traj_pts = np.ones((self.nb_points,1)) * noise_pos if noise_pos.shape[0] == 1 else noise_pos
		"""
		
		src_traj_pts = cfg['src_traj_pts']
		noise_pos = cfg['noise_pos']
		noise_pos = np.expand_dims(noise_pos, axis=0) if len(noise_pos.shape) == 1 else noise_pos
		noise_traj_pts = noise_pos

		src_timestamps, t, src_trajectory = self.get_timestamps(src_traj_pts, sph_len)
		noise_timestamps, t_noi, noise_trajectory = self.get_timestamps(noise_traj_pts, sph_len)

	
		dbg_print(f'src: {src_traj_pts}, noise: {noise_pos}\n')
		#breakpoint()

		T60 = cfg['t60']
		SNR = cfg['snr']

		src_azimuth = np.degrees(cart2sph(src_traj_pts - array_pos)[:,2])
		src_azimuth_keys = np.round(np.where(src_azimuth<0, 360+src_azimuth, src_azimuth)).astype('int32')	

		source_rirs, dp_source_rirs = self.rir_interface.get_rirs(t60=T60, idx_list=list(src_azimuth_keys))

		noi_azimuth = np.degrees(cart2sph(noise_pos - array_pos)[:,2])
		noi_azimuth_keys = np.round(np.where(noi_azimuth<0, 360+noi_azimuth, noi_azimuth)).astype('int32')	
		noise_rirs, _ = self.rir_interface.get_rirs(t60=T60, idx_list=list(noi_azimuth_keys))

		sph_reverb = self.simulate_source(sph[0].numpy(), source_rirs, src_timestamps, t)
		sph_dp = self.simulate_source(sph[0].numpy(), dp_source_rirs, src_timestamps, t)
		noi_reverb = self.simulate_source(noi[0].numpy(), noise_rirs, noise_timestamps, t)

		mic_signals, dp_signals = self.adjust_to_snr( sph_reverb, sph_dp, noi_reverb, SNR)

		DOA = cart2sph(src_trajectory - array_pos)  #[:,1:3], 
		
		mic_signals = torch.from_numpy(mic_signals.T)
		dp_signals = torch.from_numpy(dp_signals.T)
		#noise_reverb = torch.from_numpy(noise_reverb.T)


		if self.transforms is not None:
			for t in self.transforms:
				mic_signals, dp_signals, doa = t(mic_signals, dp_signals, DOA)
			return mic_signals, dp_signals, doa #, noise_reverb                 # noise_reverb is time domain signal: just for listening 

		return mic_signals, dp_signals, DOA    #, noise_reverb - for debug only
		

	def get_timestamps(self, traj_pts, sig_len):
		nb_points = traj_pts.shape[0]
		
		# Interpolate trajectory points
		timestamps = np.arange(nb_points) * sig_len / self.fs / nb_points
		t = np.arange(sig_len)/self.fs
		trajectory = np.array([np.interp(t, timestamps, traj_pts[:,i]) for i in range(3)]).transpose()

		return timestamps, t, trajectory

	def simulate_source(self, signal, RIRs, timestamps, t):
		breakpoint()
		reverb_signals = gpuRIR.simulateTrajectory(signal, RIRs, timestamps=timestamps, fs=self.fs)
		reverb_signals = reverb_signals[0:len(t),:]
		return reverb_signals

	def adjust_to_snr(self, mic_signals, dp_signals, noise_reverb, SNR):
		scale_noi = np.sqrt(np.sum(mic_signals[:,0]**2) / (np.sum(noise_reverb[:,0]**2) * (10**(SNR/10))))
		mic_signals = mic_signals + noise_reverb * scale_noi

		# normalize the root mean square of the mixture to a constant
		sph_len = mic_signals.shape[0]#*mic_signals.shape[1]          #All mics

		c = 1.0 * np.sqrt(sph_len / (np.sum(mic_signals[:,0]**2) + 1e-8))

		mic_signals *= c
		dp_signals *= c

		return mic_signals, dp_signals


if __name__=="__main__":

	logs_dir = '../signals/'
	snr = -5
	t60 = 0.2
	scenario = 'static' #'motion' 
	dataset_file = f'../dataset_file_circular_{scenario}_snr_{snr}_t60_{t60}.txt' # 'dataset_file_10sec.txt'
	train_dataset = MovingSourceDataset(dataset_file, array_setup_10cm_2mic, size =5, transforms=None) #[NetworkInput(320, 160, 0)]) #
	#breakpoint()
	train_loader = DataLoader(train_dataset, batch_size = 1, num_workers=0)
	for _batch_idx, val in enumerate(train_loader):
		print(f"mix_sig: {val[0].shape}, {val[0].dtype}, {val[0].device} \
		        tgt_sig: {val[1].shape}, {val[1].dtype}, {val[1].device} \
				doa: {val[2].shape}, {val[2].dtype}, {val[2].device} \n")

		#torchaudio.save(f'{logs_dir}sig_{scenario}_{_batch_idx}.wav', val[0][0].to(torch.float32), 16000)
		#torchaudio.save(f'{logs_dir}tgt_{scenario}_{_batch_idx}.wav', val[1][0].to(torch.float32), 16000)

		#break