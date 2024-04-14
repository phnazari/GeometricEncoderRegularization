#!/bin/bash

gpu=2

for model in 'confae_log'   # 'geomae'  # 'irae' 'geomae' 'ae' # topoae
do 
for dataset in 'celegans' 'pbmc'   # 'zilionis' 'pbmc'  'mnist' 'earth'
do
bash scripts/${dataset}/run_${model}.sh $gpu
done
done
