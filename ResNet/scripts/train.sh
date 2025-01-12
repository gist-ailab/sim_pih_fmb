#!/bin/bash
EXP_NAME='EXP_NAME'
export PROJECT_HOME="$(pwd)"
OUTPUT_DIR="PATH_TO_CHECKPOINT"
export XLA_PYTHON_CLIENT_PREALLOCATE='false'
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONPATH="$PYTHONPATH:$PROJECT_HOME/src"
 
python -m src.bc_main \
        --dataset_path="/media/sblee/170d6766-97d9-4917-8fc6-7d6ae84df8961/SSD2/workspaces/sim_pih_fmb/fmb_dataset_builder/custom_dataset/custom_dataset" \
        --dataset_name="custom_dataset:1.0.0" \
        --seed=24 \
        --dataset_image_keys='side_1:wrist_1:wrist_2' \
        --state_keys='tcp_pose:tcp_vel:tcp_force:tcp_torque' \
        --policy.state_injection='no_xy' \
        --last_action=False \
        --image_augmentation='none' \
        --total_steps=100000 \
        --eval_freq=200 \
        --train_ratio=0.98 \
        --batch_size=64 \
        --save_model=True \
        --lr=1e-4 \
        --weight_decay=1e-3 \
        --policy.spatial_aggregate='average' \
        --resnet_type='ResNet34' \
        --policy.share_resnet_between_views=False \
        --logger.output_dir="$OUTPUT_DIR/$EXP_NAME" \
        --logger.mode=enabled \
        --logger.prefix='FMB' \
        --logger.project="$EXP_NAME" \
        --train_gripper=False \
        --device='gpu' \
        --num_pegs=9 \
        --num_primitives=0