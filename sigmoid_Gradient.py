#-*- coding:utf-8 -*-
__author__ = 'fuxin'
from sigmoid import *
def sigmoidGradient(z):
  '''
  sigmoid的梯度函数（其实就是求导）
  '''
  g = sigmoid(z) * (1 - sigmoid(z))
  return g