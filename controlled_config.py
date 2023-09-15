from utils import Parameter
from locata_utils import *
from debug import dbg_print

import numpy as np
import random
#import gpuRIR
#from AcousticScene import AcousticScene

from collections import namedtuple
import torch
import torchaudio
import os

#Ablation studies
class ControlledConfig():
	def __init__(self, array_setup, snr, t60, src_mic_dist, nb_points, same_plane, noi_mic_dist=None):
		
		# expecting all Parameter instance inputs

		#Creating Static Dataset 
		self.snr = Parameter(snr)
		self.t60 = Parameter(t60)

		self.room_size = Parameter([8, 8, 3])
		self.array_pos = Parameter([0.5, 0.5, 0.5]) # For front and back of the array Circular motion
		
		self.abs_weights = Parameter([0.5]*6, [1.0]*6)  # Random absorption weights ratios between walls
		self.array_setup = array_setup

		self.nb_points = nb_points

		self._same_plane = same_plane

		self.src_mic_dist = Parameter(src_mic_dist)
		self.noi_mic_dist = Parameter(src_mic_dist) if noi_mic_dist is None else Parameter(noi_mic_dist)

		self.theta_grid = Parameter(0, 360) #np.radians(           #degrees -> radians
		self.int_theta_grid = True                                # allowed only integer theta in degrees



	def roundVal(self, x):
		# implemented to work only on Integer theta (in degrees)
		# To ensure moving source and stationary see the same set of RIRs
		return np.round(x) if self.int_theta_grid else x

	def get_pos(self, r, azimuth):
		"""
		Support for In plane only
		"""
		src_pos = np.array([r*np.cos(azimuth), r*np.sin(azimuth), 0.0])
		return src_pos

	def gen_circular_motion_pos(self):
		"""
		Circular motion: Non Overlaping Source Trajectory and noise
		Allowed Source angular motion: [10 - 180] degrees
		Minimun Source to Noise Angular distance: 5 degrees

		Output: Cartesian
				src_pos: [nb_points, 3] 
				noi_pos: [1, 3] 
		"""
		
		MIN_SRC_MOV = 10
		MAX_SRC_MOV = 180
		MIN_SRC_NOI_DIST = 5

		src_mic_dist = self.src_mic_dist.getValue()
		noi_mic_dist = self.noi_mic_dist.getValue()

		# Generate random theta on cirle
		src_start_theta = self.theta_grid.getValue()
		src_start_theta = self.roundVal(src_start_theta)

		src_end_theta = src_start_theta + MIN_SRC_MOV + random.random()* (MAX_SRC_MOV - MIN_SRC_MOV)    # Min src movement in terms of angle : 10 degrees
		src_end_theta = self.roundVal(src_end_theta)

		delta_src_movement = np.abs(src_end_theta - src_start_theta)
		noise_theta = src_end_theta + MIN_SRC_NOI_DIST + random.random()* (360 - delta_src_movement - MIN_SRC_NOI_DIST*2 )   # Min src to noise dist in terms of angle : 5 degrees
		noise_theta = self.roundVal(noise_theta)
		#print(f"src_start: {np.degrees(src_start_theta) % 360}, src_end: {np.degrees(src_end_theta) % 360}, noise: {np.degrees(noise_theta) % 360}")

		noise_pos = self.get_pos(noi_mic_dist, np.radians(noise_theta))

		src_theta_traj = np.linspace(src_start_theta, src_end_theta, self.nb_points)
		src_theta_traj = self.roundVal(src_theta_traj)

		src_pos = np.array( [ self.get_pos(src_mic_dist, np.radians(theta)) for theta in src_theta_traj ] )	

		noise_pos = np.expand_dims(noise_pos, axis=0)
		dbg_print(f'{src_theta_traj}')
		
		return src_pos, noise_pos

	def gen_static_scene_on_circle_pos(self):
		"""
		Circular Scene: 
		Minimun Source to Noise Angular distance: 5 degrees

		Output: src_pos: [1, 3]
				noi_pos: [1, 3]
		"""
		MIN_SRC_NOI_DIST = 5

		src_mic_dist = self.src_mic_dist.getValue()
		noi_mic_dist = self.noi_mic_dist.getValue()
	
		# Generate random theta on cirle
		src_theta = self.theta_grid.getValue()
		src_theta = self.roundVal(src_theta)
		noi_theta = src_theta + MIN_SRC_NOI_DIST + random.random()*(360 - MIN_SRC_NOI_DIST*2)   # Min src to noise dist in terms of angle : 5 degrees
		noi_theta = self.roundVal(noi_theta)

		src_pos = self.get_pos(src_mic_dist, np.radians(src_theta))
		noise_pos = self.get_pos(noi_mic_dist, np.radians(noi_theta))

		src_pos = np.expand_dims(src_pos, axis=0)
		noise_pos = np.expand_dims(noise_pos, axis=0)

		return src_pos, noise_pos

	def _create_acoustic_scene_config(self, scene_type, scenario=None):

		# Room
		room_sz = self.room_size.getValue()
		T60 = self.t60.getValue()

		#abs_weights = self.abs_weights.getValue()
		#beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights)

		# Microphones
		array_pos = self.array_pos.getValue() * room_sz
		mic_pos = array_pos + self.array_setup.mic_pos

		#source_trajectory
		if "Static" == scene_type: #random.random() < self.static_prob: 
			src_pos, noise_pos = self.gen_static_scene_on_circle_pos()

		elif "CircularMotion" == scene_type:
			if "source_moving"==scenario:
				src_pos, noise_pos = self.gen_circular_motion_pos()
			elif "noise_moving"==scenario:
				noise_pos, src_pos = self.gen_circular_motion_pos()
			else:
				print("Both Sources Moving Needs to be Implemented \n")
				

		else:
			print("Needs to be Implemented \n")
			

		#global pos
		noise_pos += array_pos
		src_pos += array_pos

		# *** assert src_pos, noise_pos inside room
		assert not ((src_pos >= room_sz).any() or (src_pos <= 0).any())
		assert not ((noise_pos >= room_sz).any() or (noise_pos <= 0).any())

		snr = self.snr.getValue()

		scene_dict = {}

		scene_dict['snr'] = snr
		scene_dict['t60'] = T60
		#scene_dict['beta'] = beta
		scene_dict['room_sz'] = room_sz
		#scene_dict['abs_weights'] = abs_weights
		scene_dict['mic_pos'] = mic_pos
		scene_dict['array_pos'] = array_pos # mic center
		scene_dict['src_traj_pts'] = src_pos
		scene_dict['noise_pos'] = noise_pos
		scene_dict['nb_points'] = src_pos.shape[0]

		return scene_dict

"""
	def simulate_acoustic_scene(self, scene_dict, source_signal, noise_signal):

		self.fs = 16000

		nb_points = scene_dict['nb_points']
		traj_pts = scene_dict['src_traj_pts']

		# Interpolate trajectory points
		timestamps = np.arange(nb_points) * len(source_signal) / self.fs / nb_points
		t = np.arange(len(source_signal))/self.fs
		trajectory = np.array([np.interp(t, timestamps, traj_pts[:,i]) for i in range(3)]).transpose()

		acoustic_scene = AcousticScene(
			room_sz = scene_dict['room_sz'],
			T60 = scene_dict['t60'],
			beta = scene_dict['beta'],
			SNR = scene_dict['snr'],
			array_setup = self.array_setup,
			mic_pos = scene_dict['mic_pos'],
			source_signal = source_signal, #sph[0],
			noise_signal = noise_signal, #noi[0],
			fs = self.fs,
			t = t,
			traj_pts = scene_dict['src_traj_pts'],
			timestamps = timestamps,
			trajectory = trajectory,
			DOA = cart2sph(trajectory - scene_dict['array_pos'])[:,1:3], 
			noise_pos = scene_dict['noise_pos'],
			noise_traj_pts = None    
		)

		src_reverb, dp_src_signal, noise_reverb = acoustic_scene.static_simulate()
		mic_signals, dp_signals = acoustic_scene.adjust_to_snr(mic_signals=src_reverb, dp_signals=dp_src_signal, noise_reverb=noise_reverb)

		return mic_signals, dp_signals, acoustic_scene
"""





if __name__=="__main__":
	from array_setup import array_setup_10cm_2mic
	snr = -5
	t60 = 0.2
	src_mic_dist = 1.0
	noi_mic_dist = 1.0
	scenario = "source_moving"
	test_config = ControlledConfig(array_setup=array_setup_10cm_2mic, snr=snr, t60=t60, src_mic_dist = src_mic_dist, noi_mic_dist = noi_mic_dist, nb_points=16, same_plane=True)

	for idx in range(0, 3000):									
		retry_flag = True
		retry_count = 0
		while(retry_flag):
			try:
				static_config_dict = test_config._create_acoustic_scene_config("Static")
				circular_motion_config_dict = test_config._create_acoustic_scene_config("CircularMotion", scenario)
				retry_flag = False
			except AssertionError:
				retry_flag = True
				retry_count += 1

		#breakpoint()
		scratch_dir='/fs/scratch/PAS0774/Shanmukh/Controlled_Experimets/'
		pp_str = f'{scratch_dir}test_configs/{scenario}/snr_{snr}_t60_{t60}/src_mic_dist_{src_mic_dist}_noi_mic_dist_{noi_mic_dist}/'

		os.makedirs(pp_str, exist_ok=True)
		#breakpoint()
		torch.save(circular_motion_config_dict, f'{pp_str}config_circular_motion_{idx}.pt')
		torch.save(static_config_dict, f'{pp_str}config_static_circular_{idx}.pt')
		print(f"Successful Execution {idx}\n")


