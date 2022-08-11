''' Prediction using the predictive models from amino acid sequence

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
from tensorflow.python.keras.callbacks import EarlyStopping
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

import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir',type=str, default='model',
    help='Base directory for the model.')

parser.add_argument(
    '--batch_size', type=int, default=4, help="Number of batches.")

parser.add_argument(
    '--test_number', type=int, default=35, help="Number of test points.")

parser.add_argument(
    '--testfile', type=str, default='allprop.csv', help='Datafile for the training.')

parser.add_argument(
    '--input_weights', type=str, default='weights.h5', help='input weights.')

parser.add_argument(
    '--output_prd_test', type=str, default='prd.dat', help='Write the predictions to a text file.')

arg = parser.parse_args()

def prepare_dist_data(testname):

    label_encoder = LabelEncoder()

    test_data = genfromtxt(testname, delimiter=',')
    test_features = test_data[:,range(test_data[0].size-2)]
    test_label = test_data[:,range(test_data[0].size-2,test_data[0].size-1)]
    test_datamut = genfromtxt(testname, delimiter=',', dtype='unicode')
    test_mutants = test_datamut[:,range(test_datamut[0].size-1,test_datamut[0].size)]

    return test_features,test_label,test_mutants

def write_results(models,
                 data,
                 batch_size=10,
                 model_name="model"):

    x_test, y_test = data
    pred_test = regressor.predict(x_test,batch_size=batch_size)
    pred_test = pred_test.tolist()

    for s in range(testnumber):
      print(mutants_test[s][0],'label',y_test[s][0],"prediction",pred_test[s][0],file=outputfile_prd_test)

    outputfile_prd_test.close()
x_test,y_test,mutants_test = prepare_dist_data(arg.testfile)
testnumber=arg.test_number
x_test = to_categorical(x_test,21)

input_shape = (31,21, )
batch_size = arg.batch_size
out_dim = 1

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

plot_model(regressor, to_file='seq_regressor.png', show_shapes=True)

os.makedirs(arg.model_dir, exist_ok=True)
outputfile_prd_test = open('%s/%s'%(arg.model_dir,arg.output_prd_test),"w")

if __name__ == '__main__':
    data = (x_test, y_test)
    models = (regressor)
    regressor.compile()
    regressor.summary()

    regressor.load_weights('%s/%s'%(arg.model_dir,arg.input_weights))

    write_results(models,
                 data,
                 batch_size=batch_size,
                 model_name=arg.model_dir)


