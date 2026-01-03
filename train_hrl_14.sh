#!/bin/bash
# Train C-ASE HRL (High-Level Controller) with hybrid action space for 14 skills
#
# The HRL learns to output:
# - skill_label c (discrete, via Gumbel-Softmax)
# - latent z (continuous, normalized to unit sphere)
#
# Prerequisites:
# 1. Train C-ASE low-level controller first: bash train_14.sh
# 2. Update --llc_checkpoint to your trained C-ASE model path

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# Available tasks: HumanoidHeading, HumanoidLocation, HumanoidReach, HumanoidStrike
python case/run.py --task HumanoidHeading \
--cfg_env case/data/cfg/humanoid_sword_shield_heading.yaml \
--cfg_train case/data/cfg/train/rlg/case_hrl_humanoid.yaml \
--motion_file case/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield_avg.yaml \
--clip2group_mapping case/data/motions/reallusion_sword_shield/motion_to_group_mapping_14skills.yaml \
--num_envs 4096 --nlabels 14 --numAMPObsSteps 20 --labellength 0 \
--llc_nlabels 14 --llc_skill_latent_size 64 --llc_style_latent_size 16 \
--llc_numAMPObsSteps 20 --llc_if_dropout --llc_dropout_rate 0.1 \
--experiment case_hrl_heading_14 \
--headless --save_frequency 10 \
--llc_checkpoint runs/case_14skills_00050000.pth