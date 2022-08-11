''' Training the predictive models from amino acid sequence 

 Michael Feig, Bercem Dutagaci
 Michigan State University
 2022

 bioRxiv:
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import LabelEncoder
from math import sqrt
import tensorflow as tf
from tensorflow.python.keras.layers import Lambda, Input, Dense, Flatten, Embedding, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.losses import mse
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import optimizers
from tensorflow.keras.optimizers import Adam, Adagrad 
import tensorflow.keras.losses
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.framework import ops
from scipy.stats import gaussian_kde
from scipy.stats import norm
import matplotlib.pyplot as plt
import argparse
import os
import scipy.stats
import math

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir',type=str, default='model',
    help='Base directory for the model.')

parser.add_argument(
    '--train_epochs', type=int, default=10, help='Number of training epochs.')

parser.add_argument(
    '--batch_size', type=int, default=100, help="Number of batches.")

parser.add_argument(
    '--train_number', type=int, default=100, help="Number of training points.")

parser.add_argument(
    '--test_number', type=int, default=35, help="Number of test points.")

parser.add_argument(
    '--optimizer', type=str, default='Adam', help='Optimizer that is either Adam or Adagrad.')

parser.add_argument(
    '--learning_rate', type=float, default=0.01, help='Learning rate.')

parser.add_argument(
    '--kl_weight', type=float, default=1.0, help='KL loss weight')

parser.add_argument(
    '--trainfile', type=str, default='allprop.csv', help='Datafile for the training set.')

parser.add_argument(
    '--testfile', type=str, default='allprop.csv', help='Datafile for the test set.')

parser.add_argument(
    '--output_loss', type=str, default='loss.dat', help='Write the loss into a text file.')

parser.add_argument(
    '--output_lossval', type=str, default='lossval.dat', help='Write the loss into a text file.')

parser.add_argument(
    '--output_weights', type=str, default='weights.h5', help='Write the weights.')

parser.add_argument(
    '--output_prd_train', type=str, default='prd.train.dat', help='Write the predictions to a text file.')

parser.add_argument(
    '--output_prd_test', type=str, default='prd.test.dat', help='Write the predictions to a text file.')

arg = parser.parse_args()

def prepare_dist_data(trainname,testname):

    label_encoder = LabelEncoder()

    train_data = genfromtxt(trainname, delimiter=',')
    train_features = train_data[:,range(train_data[0].size-2)]
    train_label = train_data[:,range(train_data[0].size-2,train_data[0].size-1)]
    train_datamut = genfromtxt(trainname, delimiter=',', dtype='unicode')
    train_mutants = train_datamut[:,range(train_datamut[0].size-1,train_datamut[0].size)]

    test_data = genfromtxt(testname, delimiter=',')
    test_features = test_data[:,range(test_data[0].size-2)]
    test_label = test_data[:,range(test_data[0].size-2,test_data[0].size-1)]
    test_datamut = genfromtxt(testname, delimiter=',', dtype='unicode')
    test_mutants = test_datamut[:,range(test_datamut[0].size-1,test_datamut[0].size)]

    return train_features,train_label,train_mutants,test_features,test_label,test_mutants

def calc_regression(x,y):
    slope, intercept = np.polyfit(x, y, 1)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    SStot = 0
    SSerr = 0
    for i in range(len(x)):
      SStot += (float(y[i])-sum(y)/len(y))**2
      SSerr += (float(y[i])-float(p(x[i])))**2
    mse = SSerr/len(y)
    R_squared = 1-(SSerr/SStot)
    return R_squared,slope,intercept,mse

def customkl(y_true, y_pred):
    pi = 3.141592653589793
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    ystd = K.std(y_true)
    ymean = K.mean(y_true)
    outstd = K.std(y_pred)
    outmean = K.mean(y_pred)
    klloss = 1/2*(K.square(ystd/outstd) + K.square(ymean-outmean)/K.square(outstd) - 1 + K.log(K.square(outstd)/K.square(ystd)))
    return klloss

def write_results(models,
                 data,
                 batch_size=10,
                 model_name="pred"):

    x_train, y_train, x_test, y_test = data

    pred_train = regressor.predict(x_train,batch_size=batch_size)
    pred_train = pred_train.tolist()
    pred_test = regressor.predict(x_test,batch_size=batch_size)
    pred_test = pred_test.tolist()
   
    for s in range(trainnumber):
      print(mutants_train[s][0],'label',y_train[s][0],"prediction",pred_train[s][0],file=outputfile_prd_train)
    for s in range(testnumber):
      print(mutants_test[s][0],'label',y_test[s][0],"prediction",pred_test[s][0],file=outputfile_prd_test)

    outputfile_prd_train.close()
    outputfile_prd_test.close()

    for ci in range(500,epochs+1,500):
      outputfile_prd_train_best = open('%s/prd.train.%s.dat'%(arg.model_dir,ci),"w")
      outputfile_prd_test_best = open('%s/prd.test.%s.dat'%(arg.model_dir,ci),"w")
      regressor.load_weights("%s/%s.hdf5"%(arg.model_dir,ci))

      pred_train = regressor.predict(x_train,batch_size=batch_size)
      pred_train = pred_train.tolist()
      pred_test = regressor.predict(x_test,batch_size=batch_size)
      pred_test = pred_test.tolist()

      for s in range(trainnumber):
        print(mutants_train[s][0],'label',y_train[s][0],"prediction",pred_train[s][0],file=outputfile_prd_train_best)
      for s in range(testnumber):
        print(mutants_test[s][0],'label',y_test[s][0],"prediction",pred_test[s][0],file=outputfile_prd_test_best)

      outputfile_prd_train_best.close()
      outputfile_prd_test_best.close()

x_train,y_train,mutants_train,x_test,y_test,mutants_test = prepare_dist_data(arg.trainfile,arg.testfile)
trainnumber=arg.train_number
testnumber=arg.test_number
x_train = to_categorical(x_train,21)
x_test = to_categorical(x_test,21)

input_shape = (31,21, )
batch_size = arg.batch_size
epochs = arg.train_epochs
out_dim = 1

# build model
d1=128
d2=64
d3=32
inputs = Input(shape=input_shape, name='input')
x = Dense(d1, activation='relu')(inputs)
x = Dense(d2, activation='relu')(x)
x = Dense(d3, activation='relu')(x)

x = Flatten()(x)
x = Dense(32, activation='relu')(x)

outputs = Dense(out_dim,dtype = tf.float32, name='output')(x)

regressor = Model(inputs, outputs, name='regressor')
regressor.summary()

plot_model(regressor, to_file='%s/mlp_seq_regressor.png'%arg.model_dir, show_shapes=True)

os.makedirs(arg.model_dir, exist_ok=True)

outputfile_prd_train = open('%s/%s'%(arg.model_dir,arg.output_prd_train),"w")
outputfile_prd_test = open('%s/%s'%(arg.model_dir,arg.output_prd_test),"w")

if __name__ == '__main__':
    data = (x_train, y_train, x_test, y_test)
    models = (regressor)
    if arg.optimizer=="Adagrad":
      opt = Adagrad(lr=arg.learning_rate)
    else:
      opt = Adam(lr=arg.learning_rate)
    regressor.compile(optimizer=opt,loss=['mse',customkl],loss_weights=[1,arg.kl_weight])
    regressor.summary()

    es = EarlyStopping(monitor='val_loss', mode='min',verbose=1, patience=500, min_delta=0.001)
    filepath="%s/{epoch:02d}.hdf5"%(arg.model_dir)
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_freq=500)
    modeltl = regressor.fit(x_train,y_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(x_test,y_test, None),callbacks=[checkpoint])
    regressor.save_weights('%s/%s'%(arg.model_dir,arg.output_weights))
    loss_history = modeltl.history["loss"]
    np_loss_history = np.array(loss_history)
    np.savetxt('%s/%s'%(arg.model_dir,arg.output_loss), np_loss_history, delimiter=",", fmt='%f')
    lossval_history = modeltl.history["val_loss"]
    np_lossval_history = np.array(lossval_history)
    np.savetxt('%s/%s'%(arg.model_dir,arg.output_lossval), np_lossval_history, delimiter=",", fmt='%f')

    write_results(models,
                 data,
                 batch_size=batch_size,
                 model_name=arg.model_dir)


