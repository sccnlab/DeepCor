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


```
```

