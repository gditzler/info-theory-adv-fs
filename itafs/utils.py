#!/usr/bin/env python 

import numpy as np 

def discretize_data(data, nbins=10):
  """Discretize a data set uniformly with as specified number of bins.
  Args:
    data (2D numpy array): Data set with n_samples \x n_features 
    nbins (int): Number of bins to discretize the data 

    Returns:
    data (2D numpy array): Discretize data set with n_samples \x n_features
  """
  data_out = data.copy()
  n_features = len(data_out[1])
  for n in range(n_features):
    bins = np.linspace(np.min(data[:, n]), np.max(data[:,n]), nbins)
    data_out[:,n]= np.digitize(data[:, n], bins)
  return data_out


