#-*- coding:utf-8 -*-
__author__ = 'fuxin'
import numpy as np
from nn_CostFunction import  *

def debugInitializeWeight(fan_out,fan_in):
  '''
  测试
  :param fan_out:
  :param fan_in:
  :return:
  '''
  temp = np.zeros((fan_out,fan_in + 1))
  size = temp.size
  W = np.sin(range(1,size+1))/10
  W.shape = (1+fan_in,fan_out)
  return W.T

def checkNNGradients(Lambda=3,input_layer_size=3,hidden_layer_size=5,num_labels=3,m=5):
  '''
  检验梯度函数的正确性，用少量数据进行测试，因为函数代价很高
  :param Lambda:
  :param input_layer_size:
  :param hidden_layer_size:
  :param num_labels:
  :param m:
  :return:
  '''
  theta1 = debugInitializeWeight(hidden_layer_size,input_layer_size)
  theta2 = debugInitializeWeight(num_labels,hidden_layer_size)
  nn_params = np.hstack((theta1.T.ravel(),theta2.T.ravel()))
  X = debugInitializeWeight(m,input_layer_size-1)
  y = 1 + np.mod(np.arange(1,m+1),num_labels)


  cost,theta1_grad,theta2_grad = nn_CostFunction(nn_params,input_layer_size,hidden_layer_size,\
                                                 num_labels,X,y,Lambda)


  numgrad = computeNumericalGradient(nn_params,input_layer_size,hidden_layer_size,\
                                     num_labels,X,y,Lambda)
  theta_grad = np.hstack((theta1_grad.T.ravel(),theta2_grad.T.ravel()))

  for i in range(len(theta_grad)):
    print numgrad[i],theta_grad[i]

  diff = np.linalg.norm(numgrad-theta_grad)/np.linalg.norm(numgrad+theta_grad)
  print diff
