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

## Reproduction Instructions
Running the jupyter notebook under the folder Simple_simulated_dataset will simulate the dataset using the trigometric functions and reproduce the training results.

To run the Realistic_data_simulation jupter notebooks, BrainIAK package is needed to be installed and the brain masks from the first participant in the Study Forest dataset are needed be downloaded from https://www.studyforrest.org/. Replace the filepath_func, filepath_gm, filepath_wm, and filepath_csf variables in the source code to be the path to the fMRI of that participants as well as probability maps of grap matter, white matter and cerebrospinal fluid respectively. 

The functional_connectivity_analysis is ran on 200 participants from ABIDE I dataset ((http ://fcon1000.projects.nitrc.org/) and the data are preprocessed. For each participant, the anatomical maps, fmri images, as well as probability maps of gm, wm, and csf are required. Moreover, the network maps from the following website (https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Yeo2011_fcMRI_clustering/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/MNI152/Centroid_coordinates/Yeo2011_7Networks_N1000.split_components.FSL_MNI152_2mm.Centroid_RAS.csv) are required to identify regions.


