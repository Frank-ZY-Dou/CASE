#!/bin/bash
# Test C-ASE HRL (High-Level Controller) for 87 skills
#
# Prerequisites:
# 1. Trained C-ASE low-level controller (train_87.sh)
# 2. Trained C-ASE HRL high-level controller (train_hrl_87.sh)

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"

# NOTE: Update checkpoint paths to your trained models
python case/run.py --test --task HumanoidHeading \
--cfg_env case/data/cfg/humanoid_sword_shield_heading.yaml \
--cfg_train case/data/cfg/train/rlg/case_hrl_humanoid.yaml \
--motion_file case/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield_avg.yaml \
--clip2group_mapping case/data/motions/reallusion_sword_shield/motion_to_group_mapping.yaml \
--num_envs 2 --nlabels 87 --numAMPObsSteps 20 --labellength 0 \
--llc_nlabels 87 --llc_skill_latent_size 64 --llc_style_latent_size 16 \
--llc_numAMPObsSteps 20 --llc_if_dropout --llc_dropout_rate 0.1 \
--llc_checkpoint runs/case_87skills_00050000.pth \
--checkpoint [your checkpoint path]