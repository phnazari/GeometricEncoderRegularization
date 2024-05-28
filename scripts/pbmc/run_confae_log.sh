#!/bin/bash

#SBATCH --job-name=pbmc_confae
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
for reg in 0.1 0.01 0.001 0.0001 0.00001 0.000001 0.0000001
do
python source/train.py \
--base_config configs/pbmc/base_config.yml \
--config configs/pbmc/confae.yml \
--logdir results2/pbmc_z2 \
--run confae-log_reg${reg}_seed${seed} \
--model.conf_reg ${reg} \
--training.seed ${seed} \
--model.reg_type conf-log \
--seed $seed \
--device $1
done
done
