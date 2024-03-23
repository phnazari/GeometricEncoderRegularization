for model in 'geomae' 'ae' 'confae_log' 'irae'  # topoae
do 
for dataset in 'zilionis' 'pbmc' 'celegans' 'mnist' 'earth'
do
sbatch scripts/${dataset}/run_${model}.sh
done
done