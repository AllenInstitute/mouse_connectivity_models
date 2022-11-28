#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=largemem
#SBATCH --mem=100000
#SBATCH --job-name=allen_institute_projection_normalization
#SBATCH --error=allen_institute_projection_normalization.err
#SBATCH --output=allen_institute_projection_normalization.log
#SBATCH --mail-type=FAIL       # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sjkoelle@gmail.com # Email to which notifications will be sent

cd /homes/sjkoelle/mouse_connectivity_models
export PATH="~/anaconda3/bin:$PATH"
source activate mouse_connectivity
python -u analyses/paper_final/modelvalidation/ELModel_LeafTarget_ProjNorm.py > ELModel_LeafTarget_ProjNorm.out 
source deactivate
echo "end"
