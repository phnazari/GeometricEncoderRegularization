#!/bin/bash

gpu=1

for dataset in 'celegans'  # 'mnist' 'earth' 'zilionis' 'pbmc'
do
for model in 'confae_log' 'geomae' 'irae' 'ae' 
do
bash scripts/${dataset}/run_${model}.sh $gpu
done
done
