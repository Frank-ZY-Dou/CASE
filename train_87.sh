#!/bin/bash
# Train C·ASE low-level controller with 87 skills

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib


CUDA_VISIBLE_DEVICES=1 python case/run.py --task HumanoidAMPGetup --cfg_env case/data/cfg/humanoid_ase_sword_shield_getup.yaml \
--cfg_train case/data/cfg/train/rlg/case_humanoid.yaml \
--motion_file case/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield_avg.yaml \
--clip2group_mapping case/data/motions/reallusion_sword_shield/motion_to_group_mapping.yaml \
--labellength 1 --num_envs 2048 --numAMPObsSteps 20 --nlabels 87 --skill_latent_size 64 --style_latent_size 16 \
--experiment case_87skills \
--if_focal --start_focal_epoch 2500 \
--dropout_rate 0.1 --if_dropout \
--headless --alpha 0.1 --save_frequency 500
