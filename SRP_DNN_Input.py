import torch
import numpy as np
import math
from masked_gcc_phat import compute_vad

class SRPDNN_features(object):
	def __init__(self, frame_len, frame_shift, array_type=None, array_setup=None):
		super().__init__()

		self.frame_len = frame_len
		self.frame_shift = frame_shift
		self.kernel = (1,frame_len)
		self.stride = (1,frame_shift)
		#self.mic_idx = ref_mic_idx   
		self.array_type = array_type
		self.array_setup = array_setup

		self.freq_range = self.frame_len//2 + 1

		pi = math.pi
		self.w = torch.from_numpy(np.array([(2*pi/self.frame_len)*k for k in range(1, self.freq_range)])).unsqueeze(dim=0).to(torch.float32)
		#print(f"w_shape: {w.shape}, u_theta: {u_theta.shape}")
		

	def __call__(self, mix_signal, tgt, DOA):  
		epsilon = 10**-10
		mix_cs = torch.stft(mix_signal, self.frame_len, self.frame_shift, self.frame_len, torch.hamming_window(self.frame_len), center=False, return_complex=True) #(2, num_freq, T)
		mix_cs = mix_cs[:,1:,:] # removing the dc component

		log_ms = torch.log(torch.abs(mix_cs) + epsilon)
		mix_ph = torch.angle(mix_cs)

		num_channels, _, _ = mix_cs.shape

		_log_ms_ph, _dp_phasediff = [], []

		sig_vad = compute_vad(tgt[0,:].cpu().numpy(), self.frame_len, self.frame_shift)

		# Code for frame_level doa
		doa = torch.from_numpy(DOA.T)
		doa = doa.unsqueeze(dim=1).unsqueeze(dim=1) #reshape required for unfold
		doa_frm = torch.nn.functional.unfold(doa, self.kernel, padding=(0,0), stride=self.stride)
		doa_frm = torch.mode(doa_frm, dim=1).values.T 

		doa_azimuth_degrees = torch.rad2deg(doa_frm[:,2])
		if "linear" in self.array_type:
			doa_azimuth_degrees = torch.abs(doa_azimuth_degrees)
		else:
			doa_azimuth_degrees = torch.where(doa_azimuth_degrees < 0, 360 + doa_azimuth_degrees, doa_azimuth_degrees)
			doa_azimuth_degrees = torch.where(doa_azimuth_degrees == 360, 0, doa_azimuth_degrees)
				 
		
		doa_frm = ((doa_azimuth_degrees + 1)*sig_vad) - 1  # -1 in doa_frm implies non-speech frames

		#breakpoint()
		u_theta = torch.stack( [torch.cos(doa_frm), torch.sin(doa_frm), torch.zeros(doa_frm.shape)], dim=1)
		u_theta = u_theta.to(torch.float32)

		mic_pos = self.array_setup.mic_pos
		
		c = 343 
		for _ch1 in range(num_channels):
			for _ch2 in range(_ch1+1, num_channels):			
				dist_mics = torch.from_numpy(mic_pos[_ch1] - mic_pos[_ch2]).unsqueeze(dim=1)
				dist_mics = dist_mics.to(dtype = torch.float32) # torch.transpose(, 0, 1)
				tou = torch.matmul(u_theta, dist_mics) / c
				#print(f"tou_shape: {tou.shape}, dist_mics: {dist_mics.shape}")
				out_lbl = torch.cat((torch.cos(torch.matmul(tou, self.w)), torch.sin(torch.matmul(tou, self.w))), dim=1)
				#print(f"out_lbl_shape: {out_lbl.shape}")
				_dp_phasediff.append(out_lbl.numpy()) #t()

				_log_ms_ph.append(torch.cat((log_ms[[_ch1]], log_ms[[_ch2]], mix_ph[[_ch1]], mix_ph[[_ch2]]), dim=0).numpy())
				
		return torch.from_numpy(np.array(_log_ms_ph)).to(dtype=torch.float32), torch.from_numpy(np.array(_dp_phasediff)).to(dtype=torch.float32), doa_frm #, acoustic_scene 