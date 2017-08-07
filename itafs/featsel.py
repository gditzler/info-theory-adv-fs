#!/usr/bin/env python 

import numpy as np 
import infotheory as it 

def mrmr(X, Y, k=5):
  """Minimum Relevancy Maximum Redundancy
  """

  selected_set = set([])
  valid_set = set([j for j in range(X.shape[1])])

  for i in range(k):
    max_score = -100000000000.
    for j in valid_set:
      rel = it.MI(X[:, j], Y)
      red = 0.
      if i != 0: 
        for k in selected_set:
          red += it.MI(X[:, j], X[:, k])
        red /= len(selected_set)
      score = rel+red
      if score > max_score:
        new_feat = j
    valid_set = valid_set.union(new_feat) 

  return np.array(list(valid_set)) 


def jmi(X, Y, k=5):
  """
  """
  return None

def jmi_adversary(X, Y, k=5):
  """
  """
  return None

def mrmr_adversary(X, Y, k=5):
  """Minimum Relevancy Maximum Redundancy with an Adversary
  """
  return None 