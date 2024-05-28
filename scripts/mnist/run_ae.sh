#!/bin/bash

#SBATCH --job-name=mnist_ae
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
--config configs/mnist/ae.yml \
--logdir results2/mnist_z2 \
--run ae_seed${seed} \
--seed $seed \
--device $1
done