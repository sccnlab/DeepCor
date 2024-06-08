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



