#!/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'
srun ~/.conda/envs/moving_source/bin/python train.py --num_nodes=$1 \
                  --num_gpu_per_node=$2 \
                  --T60=0.0 \
                  --SNR=-5.0 \
                  --dataset_dtype=stationary \
                  --dataset_condition=reverb \
                  --noise_simulation=diffuse \
                  --diffuse_files_path=/scratch/bbje/battula12/Databases/Timit/train_spk_signals \
                  --ref_mic_idx=-1 \
                  --dataset_file=../test_dataset_file_real_rir_circular_motion.txt \
                  --val_dataset_file=../test_dataset_file_real_rir_circular_motion.txt \
                  --bidirectional \
                  --batch_size=1 \
                  --num_workers=1 \
                  --ckpt_dir=/scratch/bbje/battula12/ControlledExp/random_seg/Linear_array_8cm_dp_rir_t60_0/MIMO_RI_PD/stationary \
                  --exp_name=CircularMotion \
                  --net_type=MIMO_RI_PD \
                  --model_path=epoch=98-step=4455.ckpt \
                  --input_test_filename=$3