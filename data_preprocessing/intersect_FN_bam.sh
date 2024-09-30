#!/bin/bash 

#SBATCH --job-name=interset_fn_bam
#SBATCH --output=/home/ndo/slurm_script/interset_fn_bam%j.out
#SBATCH --error=/home/ndo/slurm_script/interset_fn_bamn%j.err

#SBATCH --partition=himem
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=300G

eval "$(conda shell.bash hook)"
eval "$(/home/ndo/miniconda3/bin/conda shell.bash hook)"
conda activate /home/ndo/miniconda3/envs/notebook

#!/bin/bash

# Define paths
path1="/home/ndo/vardict_ML/all_FN"
path2="/home/ndo/vardict_ML/sample_bed"
output_path="/home/ndo/vardict_ML/true_FN"
temp_dir="/tmp/bed_chunks"

# Ensure the output directory and temporary directory exist
mkdir -p "$output_path"
mkdir -p "$temp_dir"

# Iterate over all files in path1
for file1 in "$path1"/*.bed; do
  # Extract the dataset_id from the filename in path1
  filename1=$(basename "$file1")
  dataset_id=$(echo "$filename1" | sed -E 's/all_fn_([A-Za-z0-9]+)\.bed/\1/')

  # Find the matching file in path2
  file2="$path2/${dataset_id}_bam.bed"

  # Check if the matching file exists in path2
  if [ -f "$file2" ]; then
    echo "Processing dataset_id: $dataset_id"
    
    # Split the first file into smaller chunks
    split -l 100000 "$file1" "$temp_dir/${dataset_id}_chunk_"  # Adjust chunk size based on your memory limits
    
    # Process each chunk with bedtools
    for chunk in "$temp_dir/${dataset_id}_chunk_"*; do
      bedtools intersect -a "$chunk" -b "$file2" >> "$output_path/true_FN_${dataset_id}.bed"
    done
    
    # Clean up chunks
    rm "$temp_dir/${dataset_id}_chunk_"*
    
  else
    echo "No matching file for dataset_id: $dataset_id in path2"
  fi
done

echo "Processing complete."
