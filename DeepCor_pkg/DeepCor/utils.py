#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from numpy import random

def correlation(x,y):
  """
  Calculate the correlations between two lists of time series. 
  """
  x_mean = np.repeat(x.mean(),x.shape,axis=0)
  y_mean = np.repeat(y.mean(),y.shape,axis=0)
  cov = (x-x_mean)*(y-y_mean)
  r = cov.sum()/(x.std()*y.std()*x.shape[0])
  return r

def remove_std0(arr):
  """
  From a numpy array, remove the time course which has 0 standard deviation (consistant over time).
  """
  std0_mask = np.std(arr, axis=1) != 0.0
  arr_filtered = arr[std0_mask]
  return arr_filtered, std0_mask

def compute_in(x):
  """
  Auxiliary function for compute_padding.
  """
  return (x-3)/2+1
def compute_in_size(x):
  """
  Auxiliary function for compute_padding.
  """
  for i in range(4):
    x = compute_in(x)
  return x
def compute_out_size(x):
  """
  Auxiliary function for compute_padding.
  """
  return ((((x*2+1)*2+1)*2+1)*2+1)
def compute_padding(x):
  """
  Compute the padding size for each CNN layers according to the input size.
  """
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
  return pad,int(final_size), pad_out



def r_squared_list(x,y):
  """
  calculate the R-squared between each pair of sequences in two lists of sequences
  """
  x_mean = np.repeat(np.reshape(x.mean(axis=1),(x.shape[0],1)),x.shape[1],axis=1)
  y_mean = np.repeat(np.reshape(y.mean(axis=1),(y.shape[0],1)),y.shape[1],axis=1)
  cov = (x-x_mean)*(y-y_mean)
  r_row = cov.sum(axis=1)/(x.std(axis=1)*y.std(axis=1)*x.shape[1])
  return np.square(r_row)
     

class Scaler():
    """
    Define a class that is the scaler for a input list and able to standardize the data across time.
    """
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
  """
  The customized Dataset function to store pairs of observation and ground truths, with another randomly paired noise voxels and augmented it with a beta function.
  """
  def __init__(self, X, Y, Z):
    self.obs = X
    self.gt = Y
    self.noi = Z

  def __len__(self):
    return self.gt.shape[0]

  def __getitem__(self, index):
    ### noise is multiplied by a beta coefficient
    observation = self.obs[index]
    ground_truth = self.gt[index]
    noise = self.noi[index]
    s = 2*random.beta(4,4,1)
    noise_aug = s*noise

    return observation, ground_truth, noise_aug

class TrainDataset_real(torch.utils.data.Dataset):
  """
  The customized Dataset function to store observation and another randomly paired noise voxels and augmented it with a beta function.
  """
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
  """
  The customized Dataset function to store observation only when performing denoising
  """
  def __init__(self, X):
    self.obs = X
    
  def __len__(self):
    return self.obs.shape[0]

  def __getitem__(self, index):
    observation = self.obs[index]
    return observation


class EarlyStopper:
    """
    The early stopper useed during training to avoid overfitting.
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
     

