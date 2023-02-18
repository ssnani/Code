#!/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'
srun ~/.conda/envs/moving_source/bin/python train.py --num_nodes=$1 \
                  --num_gpu_per_node=$2 \
                  --T60=0.0 \
                  --SNR=-5.0 \
                  --dataset_dtype=stationary \
                  --dataset_condition=noisy_reverb \
                  --noise_simulation=diffuse \
                  --diffuse_files_path=/scratch/bbje/battula12/Databases/Timit/train_spk_signals \
                  --ref_mic_idx=-1 \
                  --train \
                  --dataset_file=../dataset_file_circular_motion_snr_-5_t60_0.2_reverb.txt \
                  --val_dataset_file=../val_dataset_file_circular_motion_snr_-5_t60_0.2_reverb.txt \
                  --bidirectional \
                  --batch_size=64 \
                  --max_n_epochs=100 \
                  --num_workers=4 \
                  --ckpt_dir=/scratch/bbje/battula12/ControlledExp/random_seg/Linear_array_8cm_dp_rir_t60_0/ \
                  --exp_name=CircularMotion \
                  --resume_model=last.ckpt \
                  --net_type=mimo_ph_diff \
                  --array_job \
                  --input_train_filename=$3