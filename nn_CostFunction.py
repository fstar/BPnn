#-*- coding:utf-8 -*-
__author__ = 'fuxin'
import numpy as np
from sigmoid import *
from sigmoid_Gradient import *

def nn_CostFunction(theta,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda):
  '''
  cost函数（其实就是个类似方差的函数），返回J和梯度grad，理论上J应该是越来越小，因为通过梯度下降法
  越来越接近极小值
  :param theta1:初始参数
  :param input_layer_size: 输入层的大小
  :param hidden_layer_size: 隐藏层的大小
  :param num_labels: 输出层的大小
  :param X: 特征值矩阵
  :param y: label矩阵
  :param Lambda: 规格化参数Lambda
  :return:返回J和grad
  '''
  #print "theta",theta[0:hidden_layer_size * (input_layer_size + 1)]
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

  '''
  Back Propagation(BP)过程
  '''

  theta1_d = np.zeros(theta1.shape)
  theta2_d = np.zeros(theta2.shape)
  theta2_temp = theta2[:,1:]


  for i in range(m):
    delta3 = h[i] - temp[i]
    delta3.shape = (1,num_labels)

    delta2 = np.dot(delta3,theta2_temp) * sigmoidGradient(z2[i])
    delta2.shape = (1,hidden_layer_size)

    temp_Xi = X[i]
    temp_Xi.shape = (1,len(temp_Xi))
    temp_a2 = a2[i]
    temp_a2.shape = (1,len(temp_a2))



    theta1_d = theta1_d + np.dot(delta2.transpose(),temp_Xi)
    theta2_d = theta2_d + np.dot(delta3.transpose(),temp_a2)


  theta1_grad[:,0] = theta1_grad[:,0] + 1.0 / m * theta1_d[:,0]
  theta1_grad[:,1:] = theta1_grad[:,1:] + (1.0 / m * theta1_d[:,1:] + float(Lambda) / m * theta1[:,1:])


  theta2_grad[:,0] = theta2_grad[:,0] + 1.0 / m * theta2_d[:,0]
  theta2_grad[:,1:] = theta2_grad[:,1:] + (1.0 / m * theta2_d[:,1:] + float(Lambda) / m * theta2[:,1:])

  return J,theta1_grad,theta2_grad
