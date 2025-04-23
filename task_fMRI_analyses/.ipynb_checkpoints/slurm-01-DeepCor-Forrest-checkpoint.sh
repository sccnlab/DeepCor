#!/bin/bash
#SBATCH --job-name=DeepCor-Forrest-Papermill
#SBATCH --output=DeepCor-denoise-run-papermill-outputs-_%a.txt
#SBATCH --error=DeepCor-denoise-run-papermill-errors-_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=32gb
#SBATCH --partition=medium
#SBATCH --array=0-13

notebook_name='01-DeepCor-Forrest.ipynb'

analysis_name='outputs_DeepCor'
ofdir='../Results/'${analysis_name}

mkdir -p $ofdir
echo $ofdir

outname=${ofdir}/DeepCor-DeepCor-S${SLURM_ARRAY_TASK_ID}-R1.ipynb
echo $outname

papermill $notebook_name $outname --autosave-cell-every 60 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 1 -p analysis_name $analysis_name

outname=${ofdir}/DeepCor-DeepCor-S${SLURM_ARRAY_TASK_ID}-R2.ipynb
echo $outname

papermill $notebook_name $outname --autosave-cell-every 60 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 2 -p analysis_name $analysis_name

outname=${ofdir}/DeepCor-DeepCor-S${SLURM_ARRAY_TASK_ID}-R3.ipynb
echo $outname

papermill $notebook_name $outname --autosave-cell-every 60 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 3 -p analysis_name $analysis_name

outname=${ofdir}/DeepCor-DeepCor-S${SLURM_ARRAY_TASK_ID}-R4.ipynb
echo $outname

papermill $notebook_name $outname --autosave-cell-every 60 --progress-bar -p s ${SLURM_ARRAY_TASK_ID} -p r 4 -p analysis_name $analysis_name




