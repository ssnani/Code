import numpy as np
import os
import webrtcvad
import soundfile

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from debug import dbg_print

class ZQ_NetworkInput(object):
	def __init__(self, frame_len, frame_shift, ref_mic_idx=0, array_type=None):
		self.frame_len = frame_len
		self.frame_shift = frame_shift
		self.kernel = (1,frame_len)
		self.stride = (1,frame_shift)
		self.mic_idx = ref_mic_idx   
		self.array_type = array_type
		self.eps = torch.finfo(torch.float32).eps

	def __call__(self, mix, tgt, DOA):
		# Expecting shape mix, tgt: (2, *)
		# acoustic_scene has numpy objects doa (samples, 2)
		# Output shape: (2*2(r,i), T, f), (T,2)
		dbg_print(f"input : mix: {mix.shape} tgt: {tgt.shape},  doa:{DOA.shape}")
		ip_real_img = torch.stft(mix, self.frame_len, self.frame_shift, self.frame_len, torch.hamming_window(self.frame_len), center=False, return_complex=True) #(2, num_freq, T)
		tgt_real_img = torch.stft(tgt, self.frame_len, self.frame_shift, self.frame_len, torch.hamming_window(self.frame_len), center=False, return_complex=True)

		ip_real_img = torch.permute(ip_real_img, [0, 2, 1]) #(2, T, F)
		tgt_real_img = torch.permute(tgt_real_img, [0, 2, 1]) #(2, T, F)


		noise_real_img = ip_real_img - tgt_real_img

		ip_mag, ip_ph = torch.abs(ip_real_img), torch.angle(ip_real_img)
		ip_feat = torch.log10(ip_mag**2 + self.eps)    #log power spectrum

		lbl_mag, lbl_ph = torch.abs(tgt_real_img), torch.angle(tgt_real_img)
		noise_mag = torch.abs(noise_real_img)
		tgt_irm = self.get_mask(lbl_mag, noise_mag)	
		cos_pd = torch.cos(lbl_ph-ip_ph)
		tgt_feat = torch.fmax(torch.tensor(0.0), tgt_irm*cos_pd)
	

		# Code for frame_level doa
		doa = torch.from_numpy(DOA.T)
		doa = doa.unsqueeze(dim=1).unsqueeze(dim=1) #reshape required for unfold
		doa_frm = torch.nn.functional.unfold(doa, self.kernel, padding=(0,0), stride=self.stride)
		doa_frm = torch.mode(doa_frm, dim=1).values.T  

		#float32 for pytorch lightning
		ip_feat, tgt_feat, doa_frm = ip_feat.to(torch.float32), tgt_feat.to(torch.float32), doa_frm.to(torch.float32)

		#MISO 
		if -1 != self.mic_idx:       #condition for specific channels
			if self.array_type=='linear':
				tgt_feat = tgt_feat[2*self.mic_idx :2*self.mic_idx  + 2]
			else:
				#circular shift inputs trick
				tgt_feat = tgt_feat[2*self.mic_idx :2*self.mic_idx  + 2]
				#ip_format 
				#ip_feat = torch.roll(ip_feat[1:, ], mic_idx)
				
		dbg_print(f"Transform inp: {ip_feat.shape} tgt: {tgt_feat.shape},  doa:{doa_frm.shape}")
		seq_len = torch.tensor(tgt_feat.shape[1])

		ip_ri = torch.stft(mix, self.frame_len, self.frame_shift, self.frame_len, torch.hamming_window(self.frame_len), center=False, return_complex=False) #(2, num_freq, T)
		ip_ri = torch.permute(ip_ri, [0, 3, 2, 1]) #(2, 2(r,i), T, F)
		(num_ch, _ri, frms, freq) = ip_ri.shape
		ip_ri = torch.reshape(ip_ri, (num_ch*_ri, frms, freq))

		return ip_feat, tgt_feat, seq_len.repeat_interleave(tgt_feat.shape[0]), doa_frm , ip_ri
	
	def get_mask(self, lbl_mag, noise_mag):
		return torch.sqrt(lbl_mag**2/(lbl_mag**2 + noise_mag**2))   

	def get_mag_ph(self,x):
		#x : [num_mics, 2, T, F]
		return torch.sqrt(x[:,0]**2 + x[:,1]**2 + self.eps), torch.atan2(x[:,1,:,:] + self.eps, x[:,0,:,:] + self.eps)
	
