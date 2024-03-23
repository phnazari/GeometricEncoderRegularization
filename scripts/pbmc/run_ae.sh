#!/bin/bash

#SBATCH --job-name=pbmc_ae
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=30-00:00:00
#SBATCH --output %j.out
#SBATCH --error %j.err

for seed in 1 2 3 4 5
do
python source/train.py \
--base_config configs/pbmc/base_config.yml \
--config configs/pbmc/ae.yml \
--logdir results2/pbmc_z2 \
--run ae_seed${seed} \
--device cpu
done