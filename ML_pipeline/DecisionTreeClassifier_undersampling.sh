#!/bin/bash 

#SBATCH --job-name=DTC_undersampling
#SBATCH --output=/home/ndo/slurm_script/DTC_undersampling%j.out
#SBATCH --error=/home/ndo/slurm_script/DTC_undersampling%j.err

#SBATCH --partition=himem
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=20000



eval "$(conda shell.bash hook)"
eval "$(/home/ndo/miniconda3/bin/conda shell.bash hook)"
conda activate /home/ndo/miniconda3/envs/notebook

python3 /home/ndo/vardict_ML/ML_pipeline/DecisionTreeClassifier_undersampling.py