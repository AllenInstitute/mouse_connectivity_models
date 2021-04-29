#!/bin/bash
#SBATCH --job-name ELModel_LeafTarget_projNorm_v7      # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition largemem     # Slurm partition to use
#SBATCH --ntasks 16          # Number of tasks to run. By default, one CPU core will be allocated per task
#SBATCH --time 0-24:00        # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=40000     # Memory limit for each tasks (in MB)
#SBATCH -o myscript_ELModel_LeafTarget_projNorm_v7%j.out    # File to which STDOUT will be written
#SBATCH -e myscript_ELModel_LeafTarget_projNorm_v7%j.err    # File to which STDERR will be written
#SBATCH --mail-type=ALL       # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sjkoelle@gmail.com # Email to which notifications will be sent

export PATH="~/anaconda3/bin:$PATH"
source activate allen_env
python -u ELModel_LeafTarget_projNorm_v7.py > ELModel_LeafTarget_projNorm_v7.out