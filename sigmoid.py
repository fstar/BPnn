#-*- coding:utf-8 -*-
__author__ = 'fuxin'
import numpy as np
def sigmoid(z):
  '''
  sigmoid 函数
  '''
  g = 1.0 / (1.0 + np.exp(-z))
  return g