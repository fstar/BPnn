#-*- coding:utf-8 -*-
__author__ = 'fuxin'
import numpy as np
from sigmoid import *

def max_index(list):
  index = -1
  max = -1
  for i in range(len(list)):
    if list[i] > max:
      max = list[i]
      index = i
  return index + 1

def predict(theta1,theta2,X):
  m = len(X)
  num_labels = len(theta2)
  one = np.array([1] * X.shape[0],dtype="f")
  temp_X = np.column_stack((one,X))
  h1 = sigmoid(np.dot(temp_X,theta1.T))
  one = np.array([1] * h1.shape[0],dtype="f")
  temp_h1 = np.column_stack((one,h1))
  h2 = sigmoid(np.dot(temp_h1,theta2.T))
  result_list = []
  for i in range(m):
    result_list.append(max_index(h2[i]))
  return result_list
