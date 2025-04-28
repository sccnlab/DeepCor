#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 20:49:58 2023

@author: yuzhu
"""


# import libraries and files
import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel.processing as nibp
from scipy import signal
from itertools import combinations_with_replacement
from numpy import savetxt
import nibabel as nib
import math
from numpy import random
import sklearn.preprocessing  
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import List, Callable, Union, Any, TypeVar, Tuple
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn import linear_model
    
    
def correlation(x,y):
  x_mean = np.repeat(x.mean(),x.shape,axis=0)
  y_mean = np.repeat(y.mean(),y.shape,axis=0)
  cov = (x-x_mean)*(y-y_mean)
  r = cov.sum()/(x.std()*y.std()*x.shape[0])
  return r

def remove_std0(arr):
    std0 = np.argwhere(np.std(arr, axis=1) == 0.0)
    arr_o = np.delete(arr,std0 ,axis=0) 
    return arr_o

def compute_in(x):
  return (x-3)/2+1
def compute_in_size(x):
  for i in range(4):
    x = compute_in(x)
  return x
def compute_out_size(x):
  return ((((x*2+1)*2+1)*2+1)*2+1)
def compute_padding(x):
  rounding = np.ceil(compute_in_size(x))-compute_in_size(x)
  y = ((((rounding*2)*2)*2)*2)
  pad = bin(int(y)).replace('0b', '')
  if len(pad) < 4:
      for i in range(4-len(pad)):
          pad = '0' + pad
  final_size = compute_in_size(x+y)
  pad_out = bin(int(compute_out_size(final_size)-x)).replace('0b','')
  if len(pad_out) < 4:
      for i in range(4-len(pad_out)):
          pad_out = '0' + pad_out
  return pad,final_size, pad_out

class Scaler():
    def __init__(self,inputs):
        self.data = inputs
        self.mean = np.mean(inputs,axis=1)
        self.std = np.std(inputs, axis=1)
        self.vox, self.time = inputs.shape
    def transform(self,inputs):
        self.mean = np.reshape(self.mean,(self.vox,1))
        self.m_large = np.repeat(self.mean,self.time,axis=1)
        self.std = np.reshape(self.std,(self.vox,1))
        self.s_large = np.repeat(self.std,self.time,axis=1)
        return np.divide(inputs-self.m_large,self.s_large)
    def inverse_transform(self,outputs):
        return np.multiply(outputs,self.s_large)+self.m_large

class TrainDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, Y):
    self.obs = X
    self.noi = Y

  def __len__(self):
    return min(self.obs.shape[0],self.noi.shape[0])

  def __getitem__(self, index):
    observation = self.obs[index]
    noise = self.noi[index]
    s = 2*random.beta(4,4,1)
    noise_aug = s*noise
    return observation, noise_aug

class DenoiseDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X):
    self.obs = X
    
  def __len__(self):
    return self.obs.shape[0]

  def __getitem__(self, index):
    observation = self.obs[index]
    return observation


class cVAE(nn.Module):

    def __init__(self,in_channels: int,in_dim: int, latent_dim: int,hidden_dims: List = None) -> None:
        super(cVAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.in_dim = in_dim

        modules_z = []
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 256]
        
        self.pad, self.final_size, self.pad_out = compute_padding(self.in_dim)

        # Build Encoder
        for i in range(len(hidden_dims)):
            h_dim = hidden_dims[i]
            modules_z.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = int(self.pad[-i-1])),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder_z = nn.Sequential(*modules_z)
        self.fc_mu_z = nn.Linear(hidden_dims[-1]*int(self.final_size), latent_dim)
        self.fc_var_z = nn.Linear(hidden_dims[-1]*int(self.final_size), latent_dim)

        modules_s = []
        in_channels = self.in_channels
        for i in range(len(hidden_dims)):
            h_dim = hidden_dims[i]
            modules_s.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = int(self.pad[-i-1])),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder_s = nn.Sequential(*modules_s)
        self.fc_mu_s = nn.Linear(hidden_dims[-1]*int(self.final_size), latent_dim)
        self.fc_var_s = nn.Linear(hidden_dims[-1]*int(self.final_size), latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(2*latent_dim, hidden_dims[-1] * int(self.final_size))

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=int(self.pad_out[-4+i]),
                                    output_padding=int(self.pad_out[-4+i])),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose1d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=int(self.pad_out[-1]),
                                               output_padding=int(self.pad_out[-1])),
                            nn.BatchNorm1d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv1d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1))
           #out_channels

    def encode_z(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder_z(input)
  
        result = torch.flatten(result, start_dim=1)


        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu_z(result)
        log_var = self.fc_var_z(result)

        return [mu, log_var]

    def encode_s(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder_s(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu_s(result)
        log_var = self.fc_var_s(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1,256,int(self.final_size))
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward_tg(self, input: Tensor) -> List[Tensor]:
        tg_mu_z, tg_log_var_z = self.encode_z(input)
        tg_mu_s, tg_log_var_s = self.encode_s(input)
        tg_z = self.reparameterize(tg_mu_z, tg_log_var_z)
        tg_s = self.reparameterize(tg_mu_s, tg_log_var_s)
        output = self.decode(torch.cat((tg_z, tg_s),1))
        return  [output, input, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s,tg_z,tg_s]

    def forward_bg(self, input: Tensor) -> List[Tensor]:
        bg_mu_s, bg_log_var_s = self.encode_s(input)
        bg_s = self.reparameterize(bg_mu_s, bg_log_var_s)
        zeros = torch.zeros_like(bg_s)
        output = self.decode(torch.cat((zeros, bg_s),1))
        return  [output, input, bg_mu_s, bg_log_var_s]

    def forward_fg(self, input: Tensor) -> List[Tensor]:
        fg_mu_z, fg_log_var_z = self.encode_z(input)
        tg_z = self.reparameterize(fg_mu_z, fg_log_var_z)
        zeros = torch.zeros_like(tg_z)
        output = self.decode(torch.cat((tg_z, zeros),1))
        return  [output, input, fg_mu_z, fg_log_var_z]

    def loss_function(self,
                      *args,
                      ) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        beta = 0.00001
        gamma = 1

        recons_tg = args[0]
        input_tg = args[1]
        tg_mu_z = args[2]
        tg_log_var_z = args[3]
        tg_mu_s = args[4]
        tg_log_var_s = args[5]
        tg_z = args[6]
        tg_s = args[7]
        recons_bg = args[8]
        input_bg = args[9]
        bg_mu_s = args[10]
        bg_log_var_s = args[11]

        #kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons_tg, input_tg)
        recons_loss += F.mse_loss(recons_bg, input_bg)
        # recons_loss *= input_shape[0]*input_shape[1]

        # z1 = tg_z[:int(batch_size/2),:]
        # z2 = tg_z[int(batch_size/2):,:]
        # s1 = tg_s[:int(batch_size/2),:]
        # s2 = tg_s[int(batch_size/2):,:]
        # q_bar = torch.cat(torch.cat((s1,z2),1),torch.cat((s2,z1),1),0)
        # q = torch.cat(torch.cat((s1,z1),1),torch.cat((s2,z1),1),0)
        # q_bar_score = nn.Sigmoid(q_bar)
        # q_score = nn.Sigmoid(q)
        # tc_loss = torch.log(q_score/(1-q_score))
        # discriminator_loss = - torch.log(q_score) - torch.log(1-q_bar_score)

        kld_loss = 1 + tg_log_var_z - tg_mu_z ** 2 - tg_log_var_z.exp()
        kld_loss += 1 + tg_log_var_s - tg_mu_s ** 2 - tg_log_var_s.exp()
        kld_loss += 1 + bg_log_var_s - bg_mu_s ** 2 - bg_log_var_s.exp()
        kld_loss = torch.mean(-0.5 * torch.sum(kld_loss, dim = 1), dim = 0)

        loss = torch.mean(recons_loss + beta*kld_loss)
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD': kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward_fg(x)[0]
# setup the file path
filepath = '/data/anzellos/shared/transfers/Data4YU/fmriprep_ABIDE_1'
filepath_out = '/data/zhupu/FC/fca_results/new2/'
par_list = [name for name in os.listdir(filepath_out) if os.path.isdir(os.path.join(filepath_out, name))]
filepath_regions = '/data/anzellos/shared/transfers/Data4YU/YeoAtlas/Atlas_parc-7n_2mm.nii'

for i in range(50):
    par_num = par_list[i]
    print(par_num)
    filepath_outp = filepath_out + par_num 
    filepath_anat = filepath+'/'+par_num+ '/anat/'+par_num+'_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz'
    filepath_gm = filepath+'/'+par_num+ '/anat/'+par_num+'_space-MNI152NLin2009cAsym_res-2_label-GM_probseg.nii.gz'
    filepath_wm = filepath+'/'+par_num+ '/anat/'+par_num+'_space-MNI152NLin2009cAsym_res-2_label-WM_probseg.nii.gz'
    filepath_csf = filepath+'/'+par_num+ '/anat/'+par_num+'_space-MNI152NLin2009cAsym_res-2_label-CSF_probseg.nii.gz'
    filepath_func = filepath+'/'+par_num+ '/func/'+par_num+'_task-rest_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
     
    # load anatomical data
    anat = nib.load(filepath_anat)
    gm = nib.load(filepath_gm)
    wm = nib.load(filepath_wm)
    csf = nib.load(filepath_csf)
    func = nib.load(filepath_func)
    anat_regions = nib.load(filepath_regions)
    
    # transform probabiltiy map to the mask
    gm_values = gm.get_fdata()
    gm_mask = (gm_values>0.5)
    wm_values = wm.get_fdata()
    csf_values = csf.get_fdata()
    cf_values = wm_values+csf_values
    cf_mask = (cf_values>0.5)
        
    # remove the overlapped voxels
    diff = gm_mask & cf_mask
    gm_mask_c = gm_mask ^ diff
    cf_mask_c = cf_mask ^ diff
    
    # Extract functional data from masks
    func_values = func.get_fdata()[:,:,:,5:]
    func_reshaped = np.reshape(func_values,[func.shape[0]*func.shape[1]*func.shape[2],func.shape[3]-5])
    gm_reshaped = np.reshape(gm_mask_c,-1)
    cf_reshaped = np.reshape(cf_mask_c,-1)
    func_gm = func_reshaped[gm_reshaped,:] # these are the functional data in gray matter
    func_cf = func_reshaped[cf_reshaped,:] # these are the functional data in the regions of no interest
    
    #Normalization of Data
    func_gm = remove_std0(func_gm)
    func_cf = remove_std0(func_cf)
    
    obs_scale = Scaler(func_gm)
    obs_list = obs_scale.transform(func_gm)
    noi_scale = Scaler(func_cf)
    noi_list = noi_scale.transform(func_cf)
    
    # DataLoader
    train_inputs = TrainDataset(obs_list,noi_list)
    
    # dataloading 
    train_in = torch.utils.data.DataLoader(train_inputs, batch_size=64,
                                                 shuffle=True, num_workers=1)

    # cVAE model
    Tensor = TypeVar('torch.tensor')
    model = cVAE(1,func_cf.shape[1],8)
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    # Training the Model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    
    #transform the func to anatomical map
    for i in range(5,func.get_fdata().shape[3]):
      func0 = func.slicer[:,:,:,i]
      funcSize=nibp.resample_from_to(func0, anat_regions, order=1)
      if i==5:
        func_resize = np.expand_dims(funcSize.get_fdata(),axis=3)
      else:
        func_resize = np.concatenate((func_resize,np.expand_dims(funcSize.get_fdata(),axis=3)),axis=3)

    # extract the voxels from each region, and average the voxels 
    anat_mask = anat_regions.get_fdata()
    rg_reshaped = np.reshape(anat_mask,-1)
    func_rgs = []
    size_L = []
    
    # denoise the whole brain voxels
    func_reshaped = np.reshape(func_resize,[func_resize.shape[0]*func_resize.shape[1]*func_resize.shape[2],func_resize.shape[3]])
    
    ### original
    func_roi = func_reshaped[rg_reshaped!=0,:]
    func_roi_c = func_roi[np.where(np.std(func_roi, axis=1) != 0.0)]
    func_scale = Scaler(func_roi_c)
    func_in = func_scale.transform(func_roi_c)
    func_roi[np.where(np.std(func_roi, axis=1) != 0.0)] = func_in
    func_reshaped[rg_reshaped!=0,:] = func_roi
    
    func_rgs3 = []
    for i in range(1,int(anat_mask.max())+1):
        func_rg = func_reshaped[rg_reshaped==i,:]
        func_rg = remove_std0(func_rg)
        func_rg_avg = np.mean(func_rg,axis=0)
        func_rgs3.append(func_rg_avg)
    func_rg3 = np.array(func_rgs3)
    
    # apply the bandpass filter 
    func_fl3 = []
    sos = signal.butter(3, [0.01,0.1], btype='bandpass', analog=False, output='sos', fs=0.5)
    scaler3 = Scaler(func_rg3)
    func_rgs_n3 = scaler3.transform(func_rg3)
    for i in range(len(func_rgs_n3)):
        func_fl3.append(signal.sosfilt(sos, func_rgs_n3[i]))
    
    # calculate the correlation between each pair of averaged signal at different regions
    comb = combinations_with_replacement([i for i in range(51)], 2)
    fc_matrix3 = np.zeros((51,51))
    for (i,j) in comb:
        fc_matrix3[i,j] = correlation(func_fl3[i],func_fl3[j])
        
    savetxt(filepath_outp+'/'+ par_num+'_fcmatrix_nodenoise.csv', fc_matrix3, delimiter=',')

    ### deepcor
    
    denoise_inputs = DenoiseDataset(func_in)
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
    func_roi[np.where(np.std(func_roi, axis=1) != 0.0)] = outputs_all
    func_reshaped[rg_reshaped!=0,:] = func_roi    
    
    for i in range(1,int(anat_mask.max())+1):
        func_rg = func_reshaped[rg_reshaped==i,:]
        func_rg = remove_std0(func_rg)
        func_rg_avg = np.mean(func_rg,axis=0)
        func_rgs.append(func_rg_avg)
        size_L.append(func_rg.shape[0])
    
    #subtract the mean
    func_rg = np.array(func_rgs)
    size_np = np.array(size_L)
    multi = np.matmul(size_np.transpose(),func_rg)/size_np.sum()
    func_rm = np.subtract(func_rg,np.repeat(np.expand_dims(multi,axis=0), 51, axis=0))
        
    # apply the bandpass filter 
    func_fl = []
    func_fl_rm = []
    sos = signal.butter(3, [0.01,0.1], btype='bandpass', analog=False, output='sos', fs=0.5)
    scaler = Scaler(func_rg)
    func_rgs_n = scaler.transform(func_rg)
    for i in range(len(func_rgs_n)):
        func_fl.append(signal.sosfilt(sos, func_rgs_n[i]))
    scaler_rm = Scaler(func_rm)
    func_rgs_rm = scaler_rm.transform(func_rm)
    for i in range(len(func_rgs_rm)):
        func_fl_rm.append(signal.sosfilt(sos, func_rgs_rm[i]))
    
    # calculate the correlation between each pair of averaged signal at different regions
    comb = combinations_with_replacement([i for i in range(51)], 2)
    fc_matrix_rm = np.zeros((51,51))
    for (i,j) in comb:
        fc_matrix_rm[i,j] = correlation(func_fl_rm[i],func_fl_rm[j])
    
    savetxt(filepath_outp+'/'+ par_num+'_fcmatrix_DeepCor_rm.csv', fc_matrix_rm, delimiter=',')
            
    #### CompCor
    
    # PCA likes the time dimension as first. Let's transpose our data.
    func_gm_c = np.transpose(func_in)
    func_confounds_c = np.transpose(noi_list)
    # Fit PCA and extract PC timecourses
    pca = PCA(n_components = 5)
    pca.fit(func_confounds_c)
    confounds_pc = pca.fit_transform(func_confounds_c)
    confounds_pc.shape

    # linear regression on each voxel: PCs -> voxel pattern
    linear = linear_model.LinearRegression()
    linear.fit(confounds_pc, func_gm_c)
    
    # predict the activity of each voxel for this run 
    predict = linear.predict(confounds_pc)
    func_denoised = func_gm_c - predict # t x v
    func_denoised = np.transpose(func_denoised) # v x t
    func_roi[np.where(np.std(func_roi, axis=1) != 0.0)] = func_denoised
    func_reshaped[rg_reshaped!=0,:] = func_roi 
    
    func_rgs2 = []
    for i in range(1,int(anat_mask.max())+1):
        func_rg = func_reshaped[rg_reshaped==i,:]
        #func_rg = remove_std0(func_rg)
        func_rg_avg = np.mean(func_rg,axis=0)
        func_rgs2.append(func_rg_avg)
    func_rg2 = np.array(func_rgs2)
    
    # apply the bandpass filter 
    func_fl2 = []
    sos = signal.butter(3, [0.01,0.1], btype='bandpass', analog=False, output='sos', fs=0.5)
    scaler2 = Scaler(func_rg2)
    func_rgs_n2 = scaler2.transform(func_rg2)
    for i in range(len(func_rgs_n2)):
        func_fl2.append(signal.sosfilt(sos, func_rgs_n2[i]))
    
    # calculate the correlation between each pair of averaged signal at different regions
    comb = combinations_with_replacement([i for i in range(51)], 2)
    fc_matrix2 = np.zeros((51,51))
    for (i,j) in comb:
        fc_matrix2[i,j] = correlation(func_fl2[i],func_fl2[j])
        
    savetxt(filepath_outp+'/'+ par_num+'_fcmatrix_CompCor.csv', fc_matrix2, delimiter=',')
