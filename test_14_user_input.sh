#!/bin/bash
# Test C-ASE with 14 semantic skill groups
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"

# NOTE: Update --checkpoint path to your trained model
python case/run.py --test --task HumanoidAMPGetup --cfg_env case/data/cfg/humanoid_ase_sword_shield_getup_test.yaml \
--cfg_train case/data/cfg/train/rlg/case_humanoid.yaml \
--motion_file case/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield_avg.yaml \
--clip2group_mapping case/data/motions/reallusion_sword_shield/motion_to_group_mapping_14skills.yaml \
--labellength 1 --num_envs 1 --numAMPObsSteps 20 --nlabels 14 --skill_latent_size 64 --style_latent_size 16 \
--dropout_rate 0.1 --if_dropout \
--user_control \
--alpha 0.1 \
--checkpoint runs/case_14skills_00050000.pth