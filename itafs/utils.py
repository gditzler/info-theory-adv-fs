#!/usr/bin/env python 

import numpy as np 
import pandas as pd

def discretize_data(data, n_bins=10):
  """Discretize a data set uniformly with as specified number of bins.
  Args:
    data (2D numpy array): Data set with n_samples \x n_features 
    n_bins (int): Number of bins to discretize the data 

  Returns:
    data (2D numpy array): Discretize data set with n_samples \x n_features
  """
  data_out = data.copy()
  n_features = len(data_out[1])
  for n in range(n_features):
    bins = np.linspace(np.min(data[:, n]), np.max(data[:,n]), n_bins)
    data_out[:,n]= np.digitize(data[:, n], bins)
  
  return data_out

def read_file(file_name, sep=',', n_bins=None):
  """Read in a delimited file. 
  Args:
    file_name (string): Data set location 
    sep (string): file separator between columns (optional)
    n_bins (int): number of bins for the data (optional)

  Returns:
    X (2D numpy array): Data set with n_samples \x n_features
    y (numpy array): Vector of class labels
  """
  data = pd.read_csv(file_name, sep=sep, header=header)
  X = data[:,:-1]
  y = data[:,-1]

  if n_bins is not None: 
    X = discretize_data(X, n_bins)

  return X, y
