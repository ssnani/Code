#!/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'
srun ~/.conda/envs/moving_source/bin/python doa_miso_fwk_v2.py --num_nodes=$1 \
                  --num_gpu_per_node=$2 \
                  --T60=0.0 \
                  --SNR=-5.0 \
                  --dataset_dtype=stationary \
                  --dataset_condition=noisy_reverb \
                  --ref_mic_idx=-1 \
                  --dataset_file=../test_dataset_file_circular_motion_snr_-5_t60_0.2.txt \
                  --bidirectional \
                  --batch_size=1 \
                  --num_workers=1 \
                  --ckpt_dir=/scratch/bbje/battula12/ControlledExp/random_seg/stationary \
                  --exp_name=CircularMotion \
                  --model_path_0=epoch=99-step=4500.ckpt \
                  --model_path_1=epoch=99-step=4500.ckpt \
                  --input_test_filename=$3


