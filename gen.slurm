#!/bin/bash
#SBATCH --job-name=intergen            # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=2                # Number of CPU cores per task
#SBATCH --mem-per-cpu=16G                 # Memory per CPU core
#SBATCH --gres=gpu:1                     # Number of GPUs per node
#SBATCH --time 0:10:00                 # Time limit (hh:mm:ss)
#SBATCH --mail-type=begin                # Send email when job begins
#SBATCH --mail-type=end                  # Send email when job ends
#SBATCH --mail-user=ij9461@princeton.edu  # Your email



echo “Starting job execution...”                                                                                                                                                                
module purge
module load anaconda3/2024.2
conda activate intergen

cd /scratch/gpfs/ij9461/InterGen

python tools/infer.py