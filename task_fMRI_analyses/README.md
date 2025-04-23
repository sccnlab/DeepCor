# BC-ABCD-denoise
Denoising ABCD with deepcorr

Data and Code for the StudyForrest task-fMRI analyses. 

File description: 

01-DeepCor-Forrest.ipynb # Main file to train the DeepCor model
02-DeNN-Forrest.ipynb # Main file to train the DeNN model
03-analyse-task-fMRI-results.ipynb # Statistical analyses and plots
slurm-01-DeepCor-Forrest.sh # SLURM job file that uses papermill to loop 01-DeepCor-Forrest.ipynb over all subjects/runs
slurm-02-DeNN-Forrest.sh # SLURM job file that uses papermill to loop 02-DeNN-Forrest.ipynb over all subjects/runs
corr_res_forrest_DeNN.npy # Array containing FFA <-> Face Regressor correlations after DeNN denoising
corr_res_forrest_DeepCor.npy # Array containing FFA <-> Face Regressor correlations after DeepCor denosing
env_conda.txt # Dependency specification, generated using conda list > env_conda.txt
env_pip.txt # Dependency specification, generated using pip freeze > env_pip.txt
ROIs/ # Subject specific FFA ROIs used in the analyzes
mask_roi.nii # ROI mask used to train all models
mask_roni.nii # RONI mask used to train all models
