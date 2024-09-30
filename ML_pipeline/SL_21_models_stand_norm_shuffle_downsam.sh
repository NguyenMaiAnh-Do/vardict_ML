#!/bin/bash 

#SBATCH --job-name=models_21_stand_norm
#SBATCH --output=/home/ndo/slurm_script/models_21_stand_norm%j.out
#SBATCH --error=/home/ndo/slurm_script/models_21_stand_norm%j.err

#SBATCH --partition=himem
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=20000



eval "$(conda shell.bash hook)"
eval "$(/home/ndo/miniconda3/bin/conda shell.bash hook)"
conda activate /home/ndo/miniconda3/envs/notebook

python3 /home/ndo/vardict_ML/ML_pipeline/SL_21_models_stand_norm_shuffle_downsam.py