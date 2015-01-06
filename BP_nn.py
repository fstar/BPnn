#-*- coding:utf-8 -*-
__author__ = 'fuxin'
from nn_CostFunction import *
import matplotlib.pyplot as plt


def randInitializeWeights(L_in,L_out):
  '''
  随机生成初始化参数，L_in表示前一层的元素个数，L_out表示后一层的元素个数，随机数统一在[-epsilon_init,epsilon_init]
  设定 epsilon_init = sqrt(6) / sqrt(L_in + L_out)
  :param L_in: 前一层的元素个数
  :param L_out: 后一层的元素个数
  :return:返回初始化参数
  '''
  #W = np.zeros((L_out,1+L_in))
  epsilon_init = np.sqrt(6)/(np.sqrt(L_in+L_out))
  #epsilon_init = 0.12
  W = np.random.rand(L_out,1+L_in) * 2.0 * epsilon_init - epsilon_init
  return W

def BP_nn(input_layer_size,hidden_layer_size,num_labels,X,y,Lambda):
  '''
  梯度下降算法
  :param initial_nn_params:
  :param input_layer_size:
  :param hidden_layer_size:
  :param num_labels:
  :param X:
  :param y:
  :param Lambda:
  :return:
  '''
  initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size)
  initial_Theta2 = randInitializeWeights(hidden_layer_size,num_labels)
  initial_nn_params = np.hstack((initial_Theta1.T.ravel(),initial_Theta2.T.ravel()))
  params = initial_nn_params
  alpha = 2
  maxCycle = 300
  cost = 0
  log = []
  for k in range(maxCycle):

    cost,theta1_grad,theta2_grad = nn_CostFunction(params,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda)
    #print cost
    log.append(cost)
    params = params - alpha * np.hstack((theta1_grad.T.ravel(),theta2_grad.T.ravel()))
  fig = plt.figure()
  ax = fig.add_subplot(111)
  t = ax.scatter(range(maxCycle),log)
  fig.show()
  raw_input("stop")
  return cost,params
