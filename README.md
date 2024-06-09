# DeepCor: Denoising fMRI Data with Contrastive Autoencoders
This directory contains the source code for the following preprint: DeepCor: Denoising fMRI Data with Contrastive Autoencoders (https://www.biorxiv.org/content/10.1101/2023.10.31.565011v1).

## System Requirements
### Hardware Requirements
In order to run the source codes, a computer that have enough cpu memories to load the fMRI images and RMA memory to support operations is needed. GPUs are optional, will accelerate the running time, but are not necessary.

## Software Requirements
### OS Requirements
DeepCor is developed on a linux system.

### Python Dependencies
The file named requirements.txt contains the required python pacakges. In addition, BrainIAK is required to run the realistic simulations and here is the instructions from their pages to install the package: https://brainiak.org/docs/installation.html.

### Web Application
The example code is prototyped and documented in the Jupyter Notebook.

## Installation Guide
Here is the sample command to install packages into a python environment:
```
cd path-to-directory-contains-deepcor-source-code
python -m venv deepcor
source deepcor/bin/activate
pip install -r requirements.txt
```
The installation takes less than three minutes.

## Reproduction Instructions
Running the jupyter notebook under the folder Simple_simulated_dataset will simulate the dataset using the trigometric functions and reproduce the training results.

To run the Realistic_data_simulation jupter notebooks, BrainIAK package is needed to be installed and the brain masks from the first participant in the Study Forest dataset are needed be downloaded from https://www.studyforrest.org/. Replace the filepath_func, filepath_gm, filepath_wm, and filepath_csf variables in the source code to be the path to the fMRI of that participants as well as probability maps of grap matter, white matter and cerebrospinal fluid respectively. 

The functional_connectivity_analysis is ran on 200 participants from ABIDE I dataset ((http ://fcon1000.projects.nitrc.org/) and the data are preprocessed. For each participant, the anatomical maps, fmri images, as well as probability maps of gm, wm, and csf are required. Moreover, the network maps from the following website (https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/MNI152/Centroid_coordinates/Yeo2011_7Networks_N1000.split_components.FSL_MNI152_2mm.Centroid_RAS.csv) are required to identify regions.

## Demo
Here is a quick demo illustrating how to use DeepCor on your own data. Make sure that all the packages have been installed and model.py and utils.py are downloaded in the same folder as where the demo runs. Moreover, since the purpose of this demo is to denoise the gray matter in the whole brain, we didn't include the train-validation-test step here for your convenience. Again, the preprocessed fMRI data, the anatomical masks, and the probability maps for gm, wm, and csf are required to run the demo.

### Step 0 Import packages and functions
```
import os
import nibabel as nib
import numpy as np
from utils import remove_std0, Scaler, TrainDataset, DenoiseDataset
import torch
import torch.optim as optim
from typing import TypeVar
from model import cVAE
```

### Step 1 Load masks and extract the fMRI signals for ROI and RONI
Define the file path to access each masks and the output file path.
```
filepath  = '' # define the filepath to the directory
par_num  = '' # the participant number
filepath_out  = '' # the output filepath for results
filepath_anat = filepath+'/'+par_num+ '/anat/'+par_num+'_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz'
filepath_gm = filepath+'/'+par_num+ '/anat/'+par_num+'_space-MNI152NLin2009cAsym_res-2_label-GM_probseg.nii.gz'
filepath_wm = filepath+'/'+par_num+ '/anat/'+par_num+'_space-MNI152NLin2009cAsym_res-2_label-WM_probseg.nii.gz'
filepath_csf = filepath+'/'+par_num+ '/anat/'+par_num+'_space-MNI152NLin2009cAsym_res-2_label-CSF_probseg.nii.gz'
filepath_func = filepath+'/'+par_num+ '/func/'+par_num+'_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
```

Load fMRI, anatomical data, and probability maps
```
anat = nib.load(filepath_anat)
gm = nib.load(filepath_gm)
wm = nib.load(filepath_wm)
csf = nib.load(filepath_csf)
func = nib.load(filepath_func)
```

Transform probabiltiy map to the mask with a threshold of 0.5
```
gm_values = gm.get_fdata()
gm_mask = (gm_values>0.5)
wm_values = wm.get_fdata()
csf_values = csf.get_fdata()
cf_values = wm_values+csf_values
cf_mask = (cf_values>0.5)
```

Remove the overlapped voxels to avoid data leaking
```
diff = gm_mask & cf_mask
gm_mask_c = gm_mask ^ diff
cf_mask_c = cf_mask ^ diff
```

Extract functional data from masks
```
func_values = func.get_fdata()[:,:,:,5:]
func_reshaped = np.reshape(func_values,[func.shape[0]*func.shape[1]*func.shape[2],func.shape[3]-5])
gm_reshaped = np.reshape(gm_mask_c,-1)
cf_reshaped = np.reshape(cf_mask_c,-1)
func_gm = func_reshaped[gm_reshaped,:] # these are the functional data in gray matter
func_cf = func_reshaped[cf_reshaped,:] # these are the functional data in the regions of no interest
```

### Step 2 Normalize and load the data
Normalization of Data
```
func_gm, gm_valid_mask = remove_std0(func_gm)
func_cf, _ = remove_std0(func_cf)
obs_scale = Scaler(func_gm)
obs_list = obs_scale.transform(func_gm)
noi_scale = Scaler(func_cf)
noi_list = noi_scale.transform(func_cf)
```

Dataloading
```
train_inputs = TrainDataset(obs_list,noi_list)
train_in = torch.utils.data.DataLoader(train_inputs, batch_size=64,
                                             shuffle=True, num_workers=1)
```

### Step 3 Train the model
Define the model and the optimizer
```
Tensor = TypeVar('torch.tensor')
model = cVAE(1,func_cf.shape[1],8)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
```
Train and save the model
```device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
batch_size = 64
epoch_num = 20
running_loss_L = []
running_recons_L = []
running_KLD_L = []

for epoch in range(epoch_num):  # loop over the dataset multiple times    
    running_loss = 0.0
    running_reconstruction_loss = 0.0
    running_KLD = 0.0

    # Iterate over data.
    dataloader_iter_in = iter(train_in)
    for i in range(len(train_in)):
        inputs_gm,inputs_cf = next(dataloader_iter_in)

        inputs_gm = inputs_gm.unsqueeze(1).float().to(device)
        inputs_cf = inputs_cf.unsqueeze(1).float().to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # encoder + decoder
        [outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s,tg_z,tg_x] = model.forward_tg(inputs_gm)
        [outputs_cf, inputs_cf, bg_mu_s, bg_log_var_s] = model.forward_bg(inputs_cf)
        outputs = torch.concat((outputs_gm,outputs_cf),1)
        loss = model.loss_function(outputs_gm, inputs_gm, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s,tg_z,tg_x, outputs_cf, inputs_cf, bg_mu_s, bg_log_var_s)
        # backward + optimize
        loss['loss'].backward()
        optimizer.step()
        running_loss += loss['loss']
        running_reconstruction_loss += loss['Reconstruction_Loss']
        running_KLD += loss['KLD']   

    epoch_running_loss = running_loss / (len(train_in)*2)
    epoch_running_reconstruction_loss = running_reconstruction_loss / (len(train_in)*2)
    epoch_running_KLD = running_KLD / (len(train_in)*2)
    running_loss_L.append(epoch_running_loss.cpu().detach().numpy())
    running_recons_L.append(epoch_running_reconstruction_loss.cpu().detach().numpy())
    running_KLD_L.append(epoch_running_KLD.cpu().detach().numpy())
    
torch.save(model.state_dict(), filepath_out + '/model')
```

### Step 4 Use the trained model to denoise the ROI signals
Denoise the ROI signals
```
denoise_inputs = DenoiseDataset(obs_list)
denoise_in = torch.utils.data.DataLoader(denoise_inputs, batch_size=64,
                                              shuffle=False, num_workers=1)

denoise_iter_in = iter(denoise_in)
for i in range(len(denoise_in)):
    inputs = next(denoise_iter_in)
    inputs = inputs.unsqueeze(1).float().to(device)
    func_output = model.generate(inputs)
    func_output_np = func_output.squeeze().cpu().detach().numpy()
    if i == 0:
        outputs_all = func_output_np 
    else:
        outputs_all = np.concatenate((outputs_all,func_output_np), axis = 0)
```

Map the output back to the brain masks and save
```
new_matrix = np.zeros_like(func_values)
new_matrix_reshaped = np.reshape(new_matrix, [func.shape[0]*func.shape[1]*func.shape[2], func.shape[3]-5])
gm_full_indices = np.flatnonzero(gm_reshaped)
gm_full_mask = np.zeros_like(gm_reshaped, dtype=bool)
gm_full_mask[gm_full_indices] = gm_valid_mask
new_matrix_reshaped[gm_full_mask, :] = outputs_all
new_matrix = np.reshape(new_matrix_reshaped, func_values.shape)
ni_img = nib.Nifti1Image(new_matrix, affine=np.eye(4))
nib.save(ni_img, os.path.join(filepath_out, 'denoised_func.nii.gz'))
```

### Running time
The running time of the demo on one participant takes around 10 minutes. 
