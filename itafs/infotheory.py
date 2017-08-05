#!/usr/bin/env python 

import numpy as np

delta = 1e-3

def MI(X, Y):
  """Estimate the mutual information
  """
  return H(X)-Hc(X, Y)

def H(X):
  """Estimate the Entropy of a sequence of random varriables
  Args:
    X (numpy array): vector of random variables  

  Returns:
    H (float): Entropy
  """
  pX = P(X)
  z = 0.
  for p in pX: 
    if p >= delta:
      z += p*np.log(p)/np.log(2)
  z *= -1.
  return z 

def Hc(X, Y):
  """Estimate the Conditional Entropy of a sequence of random varriables
  Args:
    X (numpy array): vector of random variables  
    Y (numpy array): vector of random variables  

  Returns:
    Hc (float): Conditional Entropy
  """
  z = 0.
  norm = 0.
  X_unique = np.unique(X)
  Y_unique = np.unique(Y)

  for x in X_unique:
    for y in Y_unique: 
      norm += np.sum(1.*(X==x) * (Y==y))

  for x in X_unique:
    for y in Y_unique: 
      p_XgY = np.sum(1.*(X==x) * (Y==y))/norm
      p_Y = P(Y)[np.where(Y_unique==y)]
      if p_Y >= delta and p_XgY >= delta:
        z += p_XgY*np.log(p_XgY/p_Y)/np.log(2)
  z *= -1.
  return z


def P(X, Y=None, y=None):
  """Estimate the empirical probability distribution of P(X)
  Args:
    X (numpy array): vector of random variables  
    Y (numpy array): vector of random variables (optional)  
    y (int): conditional term (optional)  

  Returns:
    p (float): vector of probabilities 
  """
  if Y is None: 
    X_unique = np.unique(X)
    probs = np.zeros((len(X_unique),))
    for x,n in zip(X_unique, range(len(X_unique))):
      probs[n] = np.sum(1.*(X==x))/len(X)
  else:
    Xs = X[np.where(Y==y)]
    probs = P(Xs)
  return probs 

def KL(X, Y, symmetric=False):
  """Calculate the KL Divergence
  Args:
    X (numpy array): vector of random variables  
    Y (numpy array): vector of random variables  
    symmetric (bool): symmetric? (optional)  

  Returns:
    kl (float): KL Divergence 
  """
  if symmetric is False: 
    pX = P(X)
    pY = P(Y)
    z = 0.
    for p, q in zip(pX, pY): 
      if p >= delta and q >= delta:
        z += p*np.log(p/q)/np.log(2)
  else:
    z = (KL(X, Y)+KL(Y, X))/2
  return z
