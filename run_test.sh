#!/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'
srun ~/.conda/envs/moving_source/bin/python train.py --num_nodes=$1 \
                  --num_gpu_per_node=$2 \
                  --T60=0.0 \
                  --SNR=-5.0 \
                  --dataset_dtype=stationary \
                  --dataset_condition=reverb \
                  --ref_mic_idx=-1 \
                  --dataset_file=../test_dataset_file_real_rir_circular_motion.txt \
                  --val_dataset_file=../test_dataset_file_real_rir_circular_motion.txt \
                  --bidirectional \
                  --batch_size=1 \
                  --num_workers=1 \
                  --ckpt_dir=/scratch/bbje/battula12/ControlledExp/random_seg/Linear_2mic_8cm/MIMO_RI_PD/stationary \
                  --exp_name=CircularMotion_dbg \
                  --net_type=mimo_ph_diff \
                  --model_path=epoch=99-step=4500.ckpt \
                  --input_test_filename=$3


