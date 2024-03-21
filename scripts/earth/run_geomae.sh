#!/bin/bash

#SBATCH --job-name=earth_geomae
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
for reg in 1.0 0.1 0.01 0.001 0.0001 0.00001 0.000001
do
python3 source/train.py \
--base_config configs/earth/base_config.yml \
--config configs/earth/geomae.yml \
--logdir results2/earth_z2 \
--run geomae_reg${reg}_seed${seed} \
--model.geom_reg ${reg}
done
done