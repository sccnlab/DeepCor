#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:05:33 2023

@author: yuzhu
"""


import os
import numpy as np
from numpy import savetxt

filepath = '/data/zhupu/FC/fca_results/new2'
netorder = np.loadtxt('/data/zhupu/FC/Yeo2011_7Networks_N1000.split_components.FSL_MNI152_2mm.Centroid_RAS.csv',
                 delimiter=",", dtype=str)
par_list = [name for name in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, name))]

netorder_list = {}
for i in range(1,52):
  num = netorder[i+1][0]
  region = netorder[i+1][1]
  if 'Vis' in region:
    netorder_list[i-1] = 0
  elif 'SomMot' in region:
    netorder_list[i-1] = 1
  elif 'DorsAttn' in region:
    netorder_list[i-1] = 2
  elif 'SalVentAttn' in region:
    netorder_list[i-1] = 3
  elif 'Limbic' in region:
    netorder_list[i-1] = 4
  elif 'Cont' in region:
    netorder_list[i-1] = 5
  elif 'Default' in region:
    netorder_list[i-1] = 6

none_in = []
none_out = []
compcor_in = []
compcor_out = []
deepcor_in = []
deepcor_out = []
deepcor_rm_in = []
deepcor_rm_out = []
x = len(par_list)
for par_num in par_list:
  #none
  try:
      fc_n = np.loadtxt(filepath+'/'+ par_num+'/'+ par_num+'_fcmatrix_nodenoise.csv',
                 delimiter=",", dtype=float)
  except OSError:
      x-=1
      continue
  same_none = []
  diff_none = []
  fc_n_transformed = 1/2*np.log((1+fc_n)/(1-fc_n))
  x_none = np.where(fc_n!=0.0)[0]
  y_none = np.where(fc_n!=0.0)[1]
  for i in range(1275):
    if (netorder_list[x_none[i]] == netorder_list[y_none[i]]) & (x_none[i]!=y_none[i]):
      same_none.append(fc_n_transformed[x_none[i],y_none[i]])
    elif x_none[i]!=y_none[i]:
      diff_none.append(fc_n_transformed[x_none[i],y_none[i]])
  none_in.append(np.array(same_none).mean())
  none_out.append(np.array(diff_none).mean())
  #compcor
  fc_compcorr = np.loadtxt(filepath+'/'+ par_num+'/'+ par_num+'_fcmatrix_CompCor.csv',
                 delimiter=",", dtype=float)
  same_comp = []
  diff_comp = []
  fc_compcorr_transformed = 1/2*np.log((1+fc_compcorr)/(1-fc_compcorr))
  x_comp = np.where(fc_compcorr!=0.0)[0]
  y_comp = np.where(fc_compcorr!=0.0)[1]
  for i in range(1275):
    if (netorder_list[x_comp[i]] == netorder_list[y_comp[i]]) & (x_comp[i]!=y_comp[i]):
      same_comp.append(fc_compcorr_transformed[x_comp[i],y_comp[i]])
    elif x_comp[i]!=y_comp[i]:
      diff_comp.append(fc_compcorr_transformed[x_comp[i],y_comp[i]])
  compcor_in.append(np.array(same_comp).mean())
  compcor_out.append(np.array(diff_comp).mean())
  #deepcor
  
  fc_deepcorr_rm = np.loadtxt(filepath+'/'+ par_num+'/'+ par_num+'_fcmatrix_DeepCor_rm.csv',
                 delimiter=",", dtype=float)
  same_deep_rm = []
  diff_deep_rm = []
  fc_deepcorr_transformed = 1/2*np.log((1+fc_deepcorr_rm)/(1-fc_deepcorr_rm))
  x_deep_rm = np.where(fc_deepcorr_rm!=0.0)[0]
  y_deep_rm = np.where(fc_deepcorr_rm!=0.0)[1]
  for i in range(1275):
    if (netorder_list[x_deep_rm[i]] == netorder_list[y_deep_rm[i]]) & (x_deep_rm[i]!=y_deep_rm[i]):
      same_deep_rm.append(fc_deepcorr_transformed[x_deep_rm[i],y_deep_rm[i]])
    elif x_deep_rm[i]!=y_deep_rm[i]:
      diff_deep_rm.append(fc_deepcorr_transformed[x_deep_rm[i],y_deep_rm[i]])
  deepcor_rm_in.append(np.array(same_deep_rm).mean())
  deepcor_rm_out.append(np.array(diff_deep_rm).mean())

output = np.array([none_in, none_out, compcor_in, compcor_out,deepcor_rm_in,deepcor_rm_out])
print('number of participants'+str(x))
savetxt('/data/zhupu/FC/2.19/fca_analysis_stat2.csv', output, delimiter=',')