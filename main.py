#-*- coding:utf-8 -*-
from sklearn import datasets
import scipy.io as sio
import numpy
import matplotlib.pyplot as plt
from BP_nn import *
from predict import *
import pickle
import Image

def read_file(path):
  data = sio.loadmat(path)
  return data

def split_x_y(data):
  X = data['X']
  y = data['y']
  return X,y

def display(example):
  x_temp = []
  y_temp = []
  for i in range(20):
    for j in range(20):
      if example[i*20 + j] > 0:
        x_temp.append(j)
        y_temp.append(i)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  t = ax.scatter(x_temp,y_temp)
  fig.show()
  raw_input()

def load_parameter(path):
  data = read_file(path)
  theta1 = data["Theta1"]
  theta2 = data["Theta2"]
  return theta1,theta2

def save_to_pkl(dic):
    output = open('data.pkl', 'wb')
    pickle.dump(dic, output)
    output.close()
def read_from_pkl(path="data.pkl"):
    pkl_file = open(path, 'rb')
    data1 = pickle.load(pkl_file)
    pkl_file.close()
    return data1

def img_to_txt(img):
    '''
    把读取的图片变成矩阵
    '''
    size = img.size
    img_array = img.load()
    img_matrix = [["O" for i in range(size[0])]for j in range(size[1])]

    for i in range(size[0]):
        for j in range(size[1]):
            if img_array[j,i] == 0:
                img_matrix[j][i] = 1
    return img_matrix

def read_picture(path):
  img = Image.open(path,'r')
  pic = img_to_txt(img)
  return pic

if __name__ == "__main__":
  mat_path = "ex3data1.mat"
  data = read_file(mat_path)
  #theta1,theta2 = load_parameter("ex4weights.mat")
  #nn_params = np.hstack((theta1.T.ravel(),theta2.T.ravel()))
  X,y = split_x_y(data)
  #display(X[0])
  #cost,theta1,theta2 = nn_CostFunction(nn_params,400,25,10,X,y,3)
  input_layer_size = 400
  hidden_layer_size = 25
  num_labels = 10

  cost,theta = BP_nn(input_layer_size,hidden_layer_size,num_labels,X,y,0)
  save_to_pkl(theta)
  #theta = read_from_pkl()
  #theta1 = np.reshape(theta[0:hidden_layer_size * (input_layer_size + 1)],(input_layer_size+1,hidden_layer_size)).T
  #theta2 = np.reshape(theta[(hidden_layer_size * (input_layer_size + 1)):],(hidden_layer_size+1,num_labels)).T
  #print cost
  #pic = np.array(read_picture("3.png"))
  #pic = pic.ravel()
  #X = np.array(pic)
  #print pic

  #print X

  #result = predict(theta1,theta2,pic)
  #print result
  #test = np.array([1,-0.5,0,0.5,1])
  #print sigmoidGradient(test)
  # bp_nn(400,25,10,X,y)
  #randInitializeWeights(400,25)
  #checkNNGradients(3,400,25,10,X,y)
