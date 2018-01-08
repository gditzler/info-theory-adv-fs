#!/usr/bin/env python 

import numpy as np
import ctypes as c

Tool = c.CDLL('libMIToolbox.so')


def cmi(x, y, z):
  """d
  """

  n_observations = x.shape[0]

  x = 1.0*np.array(x, order="F")
  y = 1.0*np.array(y, order="F")
  z = 1.0*np.array(z, order="F")

  function = Tool.calculateConditionalMutualInformation
  function.restype = c.c_double

  result = function( x.ctypes.data_as(c.POINTER(c.c_double)), y.ctypes.data_as(c.POINTER(c.c_double)), z.ctypes.data_as(c.POINTER(c.c_double)), c.c_int(n_observations))
  return result


def mi(x, y):
  """d
  """

  n_observations = x.shape[0]

  x = 1.0*np.array(x, order="F")
  y = 1.0*np.array(y, order="F")

  function = Tool.calculateMutualInformation
  function.restype = c.c_double

  result = function( x.ctypes.data_as(c.POINTER(c.c_double)), y.ctypes.data_as(c.POINTER(c.c_double)), c.c_int(n_observations))
  return result

def joint(x, y, z):
  """d
  """

  n_observations = x.shape[0]

  x = 1.0*np.array(x, order="F")
  y = 1.0*np.array(y, order="F")
  z = 1.0*np.array(z, order="F")

  function = Tool.mergeArrays
  function.restype = c.c_int

  result = function( x.ctypes.data_as(c.POINTER(c.c_double)), y.ctypes.data_as(c.POINTER(c.c_double)), z.ctypes.data_as(c.POINTER(c.c_double)), c.c_int(n_observations))
  return z


def entropy(data):
  """d
  """

  n_observations = data.shape[0]
  data = 1.0*np.array(data, order="F")

  function = Tool.calculateEntropy
  function.restype = c.c_double

  result = function(data.ctypes.data_as(c.POINTER(c.c_double)), c.c_int(n_observations))
  return result

def joint_entropy(x, y):
  """d
  """

  n_observations = x.shape[0]

  x = 1.0*np.array(x, order="F")
  y = 1.0*np.array(y, order="F")

  function = Tool.calculateJointEntropy
  function.restype = c.c_double

  result = function( x.ctypes.data_as(c.POINTER(c.c_double)), y.ctypes.data_as(c.POINTER(c.c_double)), c.c_int(n_observations))
  return result


def conditional_entropy(x, y):
  """d
  """

  n_observations = x.shape[0]

  x = 1.0*np.array(x, order="F")
  y = 1.0*np.array(y, order="F")

  function = Tool.calculateConditionalEntropy
  function.restype = c.c_double

  result = function( x.ctypes.data_as(c.POINTER(c.c_double)), y.ctypes.data_as(c.POINTER(c.c_double)), c.c_int(n_observations))
  return result


def check_data(data, labels):
  """d
  """

  if isinstance(data, np.ndarray) is False:
    raise Exception("data must be an numpy ndarray.")
  if isinstance(labels, np.ndarray) is False:
    raise Exception("labels must be an numpy ndarray.")

  if len(data) != len(labels):
    raise Exception("data and labels must be the same length")

  return 1.0*np.array(data, order="F"), 1.0*np.array(labels, order="F")

