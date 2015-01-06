#-*- coding:utf-8 -*-
__author__ = 'fuxin'
import numpy as np
from sigmoid import *
def Cost_Function_J(theta,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda):
  theta1 = np.reshape(theta[0:hidden_layer_size * (input_layer_size + 1)],(input_layer_size+1,hidden_layer_size)).T
  theta2 = np.reshape(theta[(hidden_layer_size * (input_layer_size + 1)):],(hidden_layer_size+1,num_labels)).T

  m = len(X)
  J = 0
  theta1_grad = np.zeros(theta1.shape)
  theta2_grad = np.zeros(theta2.shape)
  one = np.array([1] * X.shape[0],dtype="f")
  X = np.column_stack((one,X))
  z2 = np.dot(X,theta1.transpose())
  a2 = sigmoid(z2)
  one = np.array([1] * a2.shape[0],dtype="f")
  a2 = np.column_stack((one,a2))
  z3 = np.dot(a2,theta2.transpose())
  h = sigmoid(z3)

  temp = np.zeros((m,num_labels))
  for i in range(m):
    temp[i][y[i]-1] = 1


  J = -(1.0/m) * np.sum(temp * np.log(h) + (1-temp) * np.log(1-h))
  #print "theta1",theta1[0][1:]
  theta1_temp = np.sum(np.array([i[1:] for i in theta1]) ** 2)
  theta2_temp = np.sum(np.array([i[1:] for i in theta2]) ** 2)
  bias = Lambda/(2.0 * m) * (theta1_temp + theta2_temp)
  J = J + bias
  return J

def computeNumericalGradient(theta,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda):
  numgrad = np.zeros(theta.shape)
  perturb = np.zeros(theta.shape)
  a = 0.0001
  for p in range(0,theta.size):
    perturb[p] = a
    loss1 = Cost_Function_J(theta-perturb,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda)
    loss2 = Cost_Function_J(theta+perturb,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda)
    numgrad[p] = (loss2 - loss1) / (2 * a)
    perturb[p] = 0
  return numgrad