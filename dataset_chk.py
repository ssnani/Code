import gpuRIR
from controlled_config import ControlledConfig
from rir_interface import taslp_RIR_Interface
from array_setup import array_setup_10cm_2mic
from locata_utils import cart2sph


import numpy as np

fs = 16000

def get_timestamps(self, traj_pts, sig_len):
	nb_points = traj_pts.shape[0]
	
	# Interpolate trajectory points
	timestamps = np.arange(nb_points) * sig_len / fs / nb_points
	t = np.arange(sig_len)/fs
	trajectory = np.array([np.interp(t, timestamps, traj_pts[:,i]) for i in range(3)]).transpose()

	return timestamps, t, trajectory

def simulate_source(signal, RIRs, t):
	reverb_signals = gpuRIR.simulateTrajectory(signal, RIRs, timestamps=timestamps, fs=fs)
	reverb_signals = reverb_signals[0:len(t),:]
	return reverb_signals



if __name__=="__main__":
	rir_interface = taslp_RIR_Interface()
	
	snr = -5
	t60 = 0.2
	src_mic_dist = 1.0
	noi_mic_dist = 1.0
	scenario = "source_moving"
	test_config = ControlledConfig(array_setup=array_setup_10cm_2mic, snr=snr, t60=t60,
									 src_mic_dist = src_mic_dist, noi_mic_dist = noi_mic_dist, 
									 nb_points=16, same_plane=True)

	for idx in range(0, 1):									
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

	config_dict = circular_motion_config_dict
	room_sz = config_dict['room_sz']
	array_pos = config_dict['array_pos']
	mic_pos = config_dict['mic_pos']
	traj_pts= config_dict['src_traj_pts']
	noise_pos= config_dict['noise_pos']
	T60= config_dict['t60']
	SNR= config_dict['snr']		

	### Mapping config pos to keys of rir (Interface)

	src_azimuth = np.degrees(cart2sph(traj_pts - array_pos)[:,2])
	src_azimuth_keys = np.round(np.where(src_azimuth<0, 360+src_azimuth, src_azimuth)).astype('int32')	

	source_rirs, dp_source_rirs = rir_interface.get_rirs(t60=T60, idx_list=list(src_azimuth_keys))

	noi_azimuth = np.degrees(cart2sph(noise_pos - array_pos)[:,2])
	noi_azimuth_keys = np.round(np.where(noi_azimuth<0, 360+noi_azimuth, noi_azimuth)).astype('int32')	
	noise_rirs, _ = rir_interface.get_rirs(t60=T60, idx_list=list(noi_azimuth_keys))

	timestamps, t, _ = get_timestamps(traj_pts, 64000)
	noise_timestamps, _, _ = get_timestamps(noise_pos, 64000)

	x = np.random.rand(64000)
	sph_reverb = simulate_source(x, source_rirs, t)
	sph_dp = simulate_source(x, dp_source_rirs, t)

	noi_reverb = simulate_source(x, noise_rirs, t)


	breakpoint()
	
	print("Succesful Execution\n")

