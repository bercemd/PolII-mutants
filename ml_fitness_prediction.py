''' Prediction using the predictive models from fitness scores


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
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.python.keras.layers import Lambda, Input, Dense, Flatten, Embedding, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.losses import mse
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras import backend as K
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
    '--input_dim', type=int, default=63, help="Dimension of input parameter tensor.")

parser.add_argument(
    '--testfile', type=str, default='all_fitness.txt', help='Datafile.')

parser.add_argument(
    '--output_prd', type=str, default='prd.dat', help='Write the predictions to a text file.')
parser.add_argument(
    '--input_weights', type=str, default='weights.h5', help='Input weights.')

arg = parser.parse_args()

def prepare_dist_data(testname):
    label_encoder = LabelEncoder()

    test_data = genfromtxt(testname, delimiter='\t') 
    test_features = test_data[1:,range(1,test_data[0].size-2)]
    test_datalab = genfromtxt(testname, delimiter='\t', dtype='unicode')
    test_mutants = test_datalab[1:,range(1)]
    test_label = test_datalab[1:,range(test_datalab[0].size-1,test_datalab[0].size)]
    return test_features,test_mutants

def predictions(models,
                test_mutants,
                testdata,
                model_name="model"):
    x_test = testdata
    predtest = models.predict(x_test)
    predtest = predtest.tolist()
    for f in range(len(predtest)):
        print(f+1,test_mutants[f][0],predtest[f][0],file=outputfile_prd)
    outputfile_prd.close()
x_test,test_mutants = prepare_dist_data(arg.testfile)
imputed_x_test = imputer.fit_transform(x_test)
original_dim = arg.input_dim
input_shape = (original_dim,)

inputs = Input(shape=input_shape, name='input')

x = Dense(256, activation='relu')(inputs)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)

outputs = Dense(1,dtype=tf.float32,name='reg_output')(x)
model = Model(inputs, outputs, name='regressor')
model.summary()

os.makedirs(arg.model_dir, exist_ok=True)

if __name__ == '__main__':
    model.compile()
    model.load_weights('%s/%s'%(arg.model_dir,arg.input_weights))
    testdata = (imputed_x_test)
    outputfile_prd=open("%s/%s"%(arg.model_dir,arg.output_prd),"w")
    predictions(model,
                 test_mutants,
                 testdata,
                 model_name=arg.model_dir)



