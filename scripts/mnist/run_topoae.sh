#!/bin/bash

#SBATCH --job-name=mnist_topoae
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=30-00:00:00
#SBATCH --output %j.out
#SBATCH --error %j.err

for seed in 1 2 3
do
python3 source/train.py \
--base_config configs/mnist/base_config.yml \
--config configs/mnist/topoae.yml \
--logdir results2/mnist_z2 \
--run topoae_reg_seed${seed} \
--decive $1
done