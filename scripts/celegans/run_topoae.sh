#!/bin/bash

#SBATCH --job-name=celegans_topoae
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=30-00:00:00
#SBATCH --output %j.out
#SBATCH --error %j.err

for seed in 1 2 3 4 5
do
python3 source/train.py \
--base_config configs/celegans/base_config.yml \
--config configs/celegans/topoae.yml \
--logdir results2/celegans_z2 \
--run topoae_reg_seed${seed}
done