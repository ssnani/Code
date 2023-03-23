import numpy as np
import os
import webrtcvad
import soundfile

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import random
from collections import namedtuple
from utils import Parameter
from locata_utils import *

from debug import dbg_print

from controlled_config import ControlledConfig
from rir_interface import taslp_RIR_Interface, taslp_real_RIR_Interface
from array_setup import get_array_set_up_from_config
from locata_utils import cart2sph
from scipy.signal import fftconvolve


GPU_RIR_IMPLEMENTATION = False
dataset_conditions = ["ideal", "noisy", "reverb", "noisy_reverb"]
dataset_dtypes = ["stationary", "moving"]

if GPU_RIR_IMPLEMENTATION:
	import gpuRIR

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

	def __init__(self, dataset_info_file, array_config: dict, transforms: list =None, size=None, T60=None, SNR=None, dataset_dtype=None, dataset_condition=None, train_flag=None, noise_simulation=None, diffuse_files_path=None):
		# T60, SNR : if provided during initialization is ONLY considered for training and NOT from config file
 
		with open(dataset_info_file, 'r') as f:
			self.tr_ex_list = [line.strip() for line in f.readlines()]

		self.fs = 16000

		self.array_config = array_config
		self.array_setup = array_config['array_setup']
		self.transforms = transforms if transforms is not None else None
		self.size = size

		self.rir_interface = taslp_RIR_Interface( array_config['array_type'], array_config['num_mics'], array_config['intermic_dist'], array_config['room_size'], train_flag) if "real_rirs" not in array_config \
							 else taslp_real_RIR_Interface(dist=array_config['dist'], num_mics=array_config['num_mics'])
		self.T = 4 
		self.T60 = T60      
		self.SNR = SNR
		self.dataset_dtype = dataset_dtype if dataset_dtype in dataset_dtypes else None
		self.dataset_condition = dataset_condition if dataset_condition in dataset_conditions else None
		self.train = train_flag
		self.noise_simulation=noise_simulation

		self.noi_spk_list= [os.path.join(diffuse_files_path, file) for file in os.listdir(diffuse_files_path)]
		
		#self.timit_list = 
	def __len__(self):
		return len(self.tr_ex_list) if self.size is None else self.size

	def __getitem_v1__(self, idx):
		line_info = self.tr_ex_list[idx].split(',')

		#sph
		sph, fs = torchaudio.load(line_info[0])
		noi, fs_noi = torchaudio.load(line_info[1])
		cfg = torch.load(line_info[2])

		dbg_print(cfg)
		if self.fs != fs_noi:
			noi = torchaudio.functional.resample(noi, fs_noi, self.fs)	

		if self.fs != fs:
			sph = torchaudio.functional.resample(sph, fs, self.fs)	

		#truncating to 10 sec utterances
		

		act_sph_len = sph.shape[1]
		noi_len = noi.shape[1]

		req_sph_len = self.fs*self.T

		dbg_print(f'sph: {act_sph_len}, noi: {noi_len}')
		#assert sph_len > 16000*4       #TODO: tempchange to 4 sec
		if act_sph_len < req_sph_len:
			sph = torch.cat((sph, torch.zeros(1, req_sph_len - act_sph_len)), dim=1)
		
		sph_len = sph.shape[1]
		### Introducing randomness

		sph_start_idx = random.randint(0, sph_len - req_sph_len)
		sph_end_idx = sph_start_idx + req_sph_len
		sph = sph[:,sph_start_idx:sph_end_idx]

		noi_start_idx = random.randint(0, noi_len - req_sph_len)
		noi_end_idx = noi_start_idx + req_sph_len
		noi = noi[:,noi_start_idx:noi_end_idx]

		sph_len = self.fs*self.T
		if 0:
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
		#breakpoint()
		src_traj_pts = src_traj_pts if self.dataset_dtype=="moving" else src_traj_pts[[0]]
		noise_pos = cfg['noise_pos']
		noise_pos = np.expand_dims(noise_pos, axis=0) if len(noise_pos.shape) == 1 else noise_pos
		noise_traj_pts = noise_pos

		src_timestamps, t, src_trajectory = self.get_timestamps(src_traj_pts, sph_len)
		noise_timestamps, t_noi, noise_trajectory = self.get_timestamps(noise_traj_pts, sph_len)

	
		dbg_print(f'src: {src_traj_pts}, noise: {noise_pos}\n')
		#breakpoint()
		if self.train:
			#reading from file and ignoring from launch shell script
			if self.dataset_condition == "noisy":
				SNR = float(line_info[3])    
				T60 = 0.0
			elif self.dataset_condition == "reverb":
				T60 = float(line_info[3]) 
				SNR = None
			elif self.dataset_condition == "noisy_reverb":
				SNR = float(line_info[3])    
				T60 = float(line_info[4]) 
		else:
			T60 = self.T60 if self.T60 is not None else cfg['t60']
			SNR = self.SNR if self.SNR is not None else cfg['snr']

		src_azimuth = np.degrees(cart2sph(src_traj_pts - array_pos)[:,2])
		if "real_rirs" not in self.array_config:
			src_azimuth_keys = np.round(np.where(src_azimuth<0, 360+src_azimuth, src_azimuth)).astype('int32')	
		else:
			# aachen database - 15 degree rir intervals(0:15:180)
			src_azimuth_keys = np.round((src_azimuth/15)).astype('int32')

		print(src_azimuth, src_azimuth_keys)
		source_rirs, dp_source_rirs = self.rir_interface.get_rirs(t60=T60, idx_list=list(src_azimuth_keys))

		noi_azimuth = np.degrees(cart2sph(noise_pos - array_pos)[:,2])
		if  "real_rirs" not in self.array_config:
			noi_azimuth_keys = np.round(np.where(noi_azimuth<0, 360+noi_azimuth, noi_azimuth)).astype('int32')	
		else:
			# aachen database - 15 degree rir intervals(0:15:180)
			noi_azimuth_keys = np.round((noi_azimuth/15)).astype('int32')
		#print(noi_azimuth, noi_azimuth_keys)
		
		noise_rirs, _ = self.rir_interface.get_rirs(t60=T60, idx_list=list(noi_azimuth_keys))

		dbg_print(f'src: {src_azimuth}, noise: {noi_azimuth}\n')
		if GPU_RIR_IMPLEMENTATION:
			sph_reverb = self.simulate_source_gpuRIR(sph[0].numpy(), source_rirs, src_timestamps, t)
			sph_dp = self.simulate_source_gpuRIR(sph[0].numpy(), dp_source_rirs, src_timestamps, t)
			noi_reverb = self.simulate_source_gpuRIR(noi[0].numpy(), noise_rirs, noise_timestamps, t)

			mic_signals, dp_signals = self.adjust_to_snr_gpuRIR( sph_reverb, sph_dp, noi_reverb, SNR)

			mic_signals = torch.from_numpy(mic_signals.T)
			dp_signals = torch.from_numpy(dp_signals.T)
			#noise_reverb = torch.from_numpy(noise_reverb.T)
		else:
			sph_reverb = self.simulate_source(sph, source_rirs, src_timestamps, t) #torch.from_numpy(source_rirs)
			sph_dp = self.simulate_source(sph, dp_source_rirs, src_timestamps, t)
			
			if self.dataset_condition == "noisy_reverb" or self.dataset_condition == "noisy":
				noi_reverb = self.simulate_source(noi, noise_rirs, noise_timestamps, t)
				mic_signals, dp_signals = self.adjust_to_snr( sph_reverb, sph_dp, noi_reverb, SNR)
			elif self.dataset_condition == "reverb" or self.dataset_condition == "ideal":
				mic_signals, dp_signals = self.rms_normalize(sph_reverb, sph_dp)

			else:
				print(f"Invalid dataset_condition: {self.dataset_condition}")


		DOA = cart2sph(src_trajectory - array_pos)  #[:,1:3], 

		if self.transforms is not None:
			for t in self.transforms:
				mic_signals, dp_signals, doa = t(mic_signals, dp_signals, DOA)
			return mic_signals, dp_signals, doa #, noise_reverb                 # noise_reverb is time domain signal: just for listening 

		return mic_signals, dp_signals, DOA    #, noise_reverb - for debug only
	
	def __getitem__(self, idx):
		line_info = self.tr_ex_list[idx].split(',')

		cfg = torch.load(line_info[2])
		dbg_print(cfg)
		mic_pos = cfg['mic_pos']
		array_pos = cfg['array_pos']

		if self.train:
			#reading from file and ignoring from launch shell script
			if self.dataset_condition == "noisy":
				SNR = float(line_info[3])    
				T60 = 0.0
			elif self.dataset_condition == "reverb":
				T60 = float(line_info[3]) 
				SNR = None
			elif self.dataset_condition == "noisy_reverb":
				SNR = float(line_info[3])    
				T60 = float(line_info[4]) 
		else:
			T60 = self.T60 if self.T60 is not None else cfg['t60']
			SNR = self.SNR if self.SNR is not None else cfg['snr']


		#sph
		sph, fs = torchaudio.load(line_info[0])
		if self.fs != fs:
			sph = torchaudio.functional.resample(sph, fs, self.fs)	

		sph = self.get_random_seg(sph,self.fs*self.T)
		sph_len = sph.shape[1]

		src_traj_pts = cfg['src_traj_pts'] 
		src_traj_pts = src_traj_pts if self.dataset_dtype=="moving" else src_traj_pts[[0]]

		src_azimuth = np.degrees(cart2sph(src_traj_pts - array_pos)[:,2])

		source_rirs, dp_source_rirs = self.get_rirs(src_azimuth, T60)

		sph_reverb, src_trajectory = self.gen_signal(sph, src_traj_pts, source_rirs)
		sph_dp, _ = self.gen_signal(sph, src_traj_pts, dp_source_rirs)

		#Noise simulations
		if "noisy" in self.dataset_condition:
			if "point_source"==self.noise_simulation:
				noi, fs_noi = torchaudio.load(line_info[1])
				if self.fs != fs_noi:
					noi = torchaudio.functional.resample(sph, fs_noi, self.fs)	
				noi = self.get_random_seg(noi,self.fs*self.T)

				noise_pos = cfg['noise_pos']
				noise_pos = np.expand_dims(noise_pos, axis=0) if len(noise_pos.shape) == 1 else noise_pos
				noise_traj_pts = noise_pos

				noi_azimuth = np.degrees(cart2sph(noise_pos - array_pos)[:,2])
				noise_rirs, _ = self.get_rirs(noi_azimuth, T60)

				noi_reverb, _ = self.gen_signal(noi, noise_traj_pts, noise_rirs)
			else:
				src_theta, nb_noise_points, res = (src_azimuth, 72, 5) if "real_rirs" not in self.array_config else (0, 13, 15)
				noi_reverb = self.gen_diffuse_noise_signals(T60, src_theta, nb_noise_points, res)
			

		if self.dataset_condition == "noisy_reverb" or self.dataset_condition == "noisy":
			if GPU_RIR_IMPLEMENTATION:

				mic_signals, dp_signals = self.adjust_to_snr_gpuRIR( sph_reverb, sph_dp, noi_reverb, SNR)

				mic_signals = torch.from_numpy(mic_signals.T)
				dp_signals = torch.from_numpy(dp_signals.T)
			else:
				mic_signals, dp_signals = self.adjust_to_snr( sph_reverb, sph_dp, noi_reverb, SNR)
		elif self.dataset_condition == "reverb" or self.dataset_condition == "ideal":
			mic_signals, dp_signals = self.rms_normalize(sph_reverb, sph_dp)
		else:
			print(f"Invalid dataset_condition: {self.dataset_condition}")
		
		DOA = cart2sph(src_trajectory - array_pos)  #[:,1:3], 

		if self.transforms is not None:
			for t in self.transforms:
				mic_signals, dp_signals, doa = t(mic_signals, dp_signals, DOA)
			return mic_signals, dp_signals, doa #, noise_reverb                 # noise_reverb is time domain signal: just for listening 

		return mic_signals, dp_signals, DOA 

	def get_timestamps(self, traj_pts, sig_len):
		nb_points = traj_pts.shape[0]
		
		# Interpolate trajectory points
		timestamps = np.arange(nb_points) * sig_len / self.fs / nb_points
		t = np.arange(sig_len)/self.fs
		trajectory = np.array([np.interp(t, timestamps, traj_pts[:,i]) for i in range(3)]).transpose()

		return timestamps, t, trajectory

	def simulate_source_gpuRIR(self, signal, RIRs, timestamps, t):	
		reverb_signals = gpuRIR.simulateTrajectory(signal, RIRs, timestamps=timestamps, fs=self.fs)
		reverb_signals = reverb_signals[0:len(t),:]
		return reverb_signals

	def simulate_source(self, signal, RIRs, timestamps, t):
		# signal: tensor( 1, sig_len), 
		# RIRs: numpy (nb_points, num_ch, rir_len)
		# reverb_signals : tensor( num_ch, sig_len)
		reverb_signals = self.simulateTrajectory(signal, RIRs, timestamps=timestamps, fs=self.fs) #torch.from_numpy(signal).unsqueeze(dim=0)
		reverb_signals = reverb_signals[:,0:len(t)]
		return reverb_signals

	def simulate_diffuse_noise(self, signals, RIRs, timestamps, t):
		# signal: tensor( nb_points, sig_len), 
		# RIRs: numpy (nb_points, num_ch, rir_len)
		# reverb_signals : tensor( num_ch, sig_len)
		nb_points = signals.shape[0]
		num_ch = RIRs.shape[1]
		reverb_signals = torch.zeros(num_ch,len(t))
		for _theta in range(nb_points):
			reverb_signals_full = self.simulateTrajectory(signals[[_theta],:], RIRs[[_theta],:,:], timestamps=timestamps, fs=self.fs) #torch.from_numpy(signal).unsqueeze(dim=0)
			reverb_signals += reverb_signals_full[:,0:len(t)]

		return reverb_signals

	def conv(self, signal, RIR):
		# signal: tensor( 1, sig_len), rir: (num_ch, rir_len)
		pad_signal = torch.nn.functional.pad(signal, (RIR.shape[1]-1, RIR.shape[1]-1))
		flip_rir = torch.flip(RIR, [1])
		reverb_signal = torch.nn.functional.conv1d(pad_signal.unsqueeze(dim=1), flip_rir.unsqueeze(dim=1))
		reverb_signal = reverb_signal.squeeze(dim=0)
		return reverb_signal	

	def fftconv(self, signal, RIR):
		# Numpy implementation
		# signal: ( 1, sig_len), rir: (num_ch, rir_len)
		
		reverb_signal = fftconvolve(signal, RIR, mode='full')
		return reverb_signal

	def get_seg(self, signal, timestamps):
		blk_len = signal.shape[-1] if len(timestamps)==1 else int(timestamps[1]*self.fs)
		seg_sig = torch.nn.functional.unfold(signal.unsqueeze(dim=1).unsqueeze(dim=1), kernel_size=(1, blk_len), padding=(0,0), stride=(1, blk_len))
		seg_sig = torch.permute(seg_sig.squeeze(dim=0),[1,0])   #(num_seg, blk_len)

		return seg_sig

	def simulateTrajectory(self, signal, RIRs, timestamps, fs):
		#signal: tensor( 1, sig_len)
		#RIRs: numpy (nb_points, num_ch, rir_len)
		
		(nb_points, num_ch, rir_len) = RIRs.shape
		nSamples = signal.shape[-1]
		w_ini = np.append((timestamps*fs).astype(int), nSamples)

		seg_signal = self.get_seg(signal, timestamps)

		reverb_signal = torch.zeros(num_ch, nSamples+rir_len-1)

		for seg_idx in range(nb_points):
			#reverb_seg = self.conv(seg_signal[[seg_idx],:], RIRs[seg_idx,:,:])
			
			reverb_seg = torch.from_numpy(self.fftconv(seg_signal[[seg_idx],:].numpy(), RIRs[seg_idx,:,:]))

			reverb_signal[:,w_ini[seg_idx] : w_ini[seg_idx+1]+rir_len-1] += reverb_seg

		return reverb_signal

	def adjust_to_snr_gpuRIR(self, mic_signals, dp_signals, noise_reverb, SNR):
		scale_noi = np.sqrt(np.sum(mic_signals[:,0]**2) / (np.sum(noise_reverb[:,0]**2) * (10**(SNR/10))))
		mic_signals = mic_signals + noise_reverb * scale_noi

		# normalize the root mean square of the mixture to a constant
		sph_len = mic_signals.shape[0]#*mic_signals.shape[1]          #All mics

		c = 1.0 * np.sqrt(sph_len / (np.sum(mic_signals[:,0]**2) + 1e-8))

		mic_signals *= c
		dp_signals *= c

		return mic_signals, dp_signals

	def adjust_to_snr(self, mic_signals, dp_signals, noise_reverb, SNR):
		# mic_signals: tensor(num_ch, sig_len)
		scale_noi = torch.sqrt(torch.sum(mic_signals[0,:]**2) / (torch.sum(noise_reverb[0,:]**2) * (10**(SNR/10))))
		mic_signals = mic_signals + noise_reverb * scale_noi

		# normalize the root mean square of the mixture to a constant
		sph_len = mic_signals.shape[1]   #*mic_signals.shape[1]          #All mics

		c = 1.0 * torch.sqrt(sph_len / (torch.sum(mic_signals[0,:]**2) + 1e-8))

		mic_signals *= c
		dp_signals *= c

		return mic_signals, dp_signals

	def rms_normalize(self, mic_signals, dp_signals):
		# normalize the root mean square of the mixture to a constant
		sph_len = mic_signals.shape[1]   #*mic_signals.shape[1]          #All mics

		c = 1.0 * torch.sqrt(sph_len / (torch.sum(mic_signals[0,:]**2) + 1e-8))

		mic_signals *= c
		dp_signals *= c
		return mic_signals, dp_signals

	def get_random_seg(self, signal, duration):
		act_sph_len = signal.shape[1]
		req_sph_len = duration

		if act_sph_len < req_sph_len:
			signal = torch.cat((signal, torch.zeros(1, req_sph_len - act_sph_len)), dim=1)
		
		sph_len = signal.shape[1]
		### Introducing randomness

		sph_start_idx = random.randint(0, sph_len - req_sph_len)
		sph_end_idx = sph_start_idx + req_sph_len
		signal = signal[:,sph_start_idx:sph_end_idx]

		return signal

	def get_rirs(self, src_azimuth, T60):

		if "real_rirs" not in self.array_config:
			src_azimuth_keys = np.round(np.where(src_azimuth<0, 360+src_azimuth, src_azimuth)).astype('int32')	
		else:
			# aachen database - 15 degree rir intervals(0:15:180)
			src_azimuth_keys = np.round((src_azimuth/15)).astype('int32')

		#print(src_azimuth, src_azimuth_keys)
		source_rirs, dp_source_rirs = self.rir_interface.get_rirs(t60=T60, idx_list=list(src_azimuth_keys))

		return source_rirs, dp_source_rirs

	def gen_diffuse_noise_signals(self, t60, theta, nb_noise_points=72, res=5):
		#generates diffuse signals of length (fs*T) from 72 directions[0,360,5(inc)]
		spk_list = random.sample(self.noi_spk_list, nb_noise_points)

		theta_list = np.array([(theta + idx*res)%360 for idx in range(nb_noise_points)])
		
		rirs, _ = self.get_rirs(theta_list[:,0], t60)
		#rirs, _ = self.rir_interface.get_rirs(t60, idx_list=theta_list)
		signals = torch.zeros(nb_noise_points,self.fs*self.T)
		for spk_idx, spk in enumerate(spk_list):
			sig, fs = torchaudio.load(spk)
			if self.fs != fs:
				sig = torchaudio.functional.resample(sig, fs, self.fs)	

			#truncating to 4 sec utterances
			signals[spk_idx,:] = self.get_random_seg(sig, self.fs*self.T)

		timestamps, t, _ = self.get_timestamps(np.zeros((1,3)), sig_len=self.fs*self.T)
		diffuse_signals = self.simulate_diffuse_noise(signals, rirs, timestamps, t)
		
		return diffuse_signals

	def gen_signal(self, signal, src_traj_pts, rirs):
		src_timestamps, t, src_trajectory = self.get_timestamps(src_traj_pts, signal.shape[1])
		if GPU_RIR_IMPLEMENTATION:
			reverb_sig = self.simulate_source_gpuRIR(signal[0].numpy(), rirs, src_timestamps, t)
		else:
			reverb_sig = self.simulate_source(signal, rirs, src_timestamps, t)

		return reverb_sig, src_trajectory

if __name__=="__main__":

	logs_dir = '../signals/'
	snr = -5
	t60 = 0.2
	scenario = 'motion' #'static' #
	dataset_file = f'../dataset_file_circular_{scenario}_snr_{snr}_t60_{t60}_reverb.txt' # 'dataset_file_10sec.txt'

	T60=0.4
	SNR=5
	dataset_dtype="stationary"
	dataset_condition="reverb"
	noise_simulation="diffuse"
	diffuse_files_path= '/fs/scratch/PAS0774/Shanmukh/Databases/Timit/train_spk_signals'
	array_config = {}

	array_config['array_type'], array_config['num_mics'], array_config['intermic_dist'], array_config['room_size'] = 'linear', 2, 8.0, [6,6,2.4]

	array_config['array_setup'] = get_array_set_up_from_config(array_config['array_type'], array_config['num_mics'], array_config['intermic_dist'])

	train_dataset = MovingSourceDataset(dataset_file, array_config, size=1, transforms=None, T60=T60, SNR=SNR, dataset_dtype=dataset_dtype, dataset_condition=dataset_condition, 
										noise_simulation=noise_simulation, diffuse_files_path= diffuse_files_path) #[NetworkInput(320, 160, 0)]) #
	#breakpoint()
	train_loader = DataLoader(train_dataset, batch_size = 1, num_workers=0)
	for _batch_idx, val in enumerate(train_loader):
		print(f"mix_sig: {val[0].shape}, {val[0].dtype}, {val[0].device} \
				tgt_sig: {val[1].shape}, {val[1].dtype}, {val[1].device} \
				doa: {val[2].shape}, {val[2].dtype}, {val[2].device} \n")

		#torchaudio.save(f'{logs_dir}{dataset_condition}_sig_{dataset_dtype}_{noise_simulation}_{_batch_idx}.wav', val[0][0].to(torch.float32), 16000)
		#torchaudio.save(f'{logs_dir}{dataset_condition}_tgt_{dataset_dtype}_{noise_simulation}_{_batch_idx}.wav', val[1][0].to(torch.float32), 16000)

		#break