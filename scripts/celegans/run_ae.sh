#!/bin/bash

#SBATCH --job-name=celegans_ae
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=30-00:00:00
#SBATCH --output %j.out
#SBATCH --error %j.err

for seed in 1 2 3
do
python source/train.py \
--base_config configs/celegans/base_config.yml \
--config configs/celegans/ae.yml \
--logdir results2/celegans_z2 \
--run ae_seed${seed} \
--decive $1
done