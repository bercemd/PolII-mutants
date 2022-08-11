''' Trained VAE model for fitness scores
 # the reference for the originial code:
   Kingma, Diederik P., and Max Welling.
   "Auto-Encoding Variational Bayes."
   https://arxiv.org/abs/1312.6114

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

import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir',type=str, default='model',
    help='Base directory for the model.')

parser.add_argument(
    '--sample_number', type=int, default=135, help="Number of samples.")

parser.add_argument(
    '--input_dim', type=int, default=62, help="Dimension of input parameter tensor.")

parser.add_argument(
    '--testfile', type=str, default='prop.fitness.csv', help='Datafile for the test.')

parser.add_argument(
    '--output_latent', type=str, default='latent.dat', help='Write the latent space variable into a text file.')

parser.add_argument(
    '--input_weights', type=str, default='weights.h5', help='Input weights.')

arg = parser.parse_args()
def prepare_dist_data(testname):

    label_encoder = LabelEncoder()

    test_data = genfromtxt(testname, delimiter=',')
    test_features = test_data[:,range(test_data[0].size-2)]
    test_label = test_data[:,range(test_data[0].size-2,test_data[0].size-1)]
    test_datamut = genfromtxt(testname, delimiter=',', dtype='unicode')
    test_mutants = test_datamut[:,range(test_datamut[0].size-1,test_datamut[0].size)]
    return test_features,test_label,test_mutants

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def write_results(models,
                 data,
                 model_name="model"):

    encoder, decoder = models
    x_test, y_test = data

    z_mean, _, _ = encoder.predict(x_test)
    for s in range(samplenumber):
      print(x_mutants[s][0],'z0',z_mean[:, 0][s],'z1',z_mean[:, 1][s],'label',y_test[s][0],file=outputfile_latent)

    outputfile_latent.close()
x_test,y_test,x_mutants = prepare_dist_data(arg.testfile)
samplenumber=arg.sample_number
imputed_x_test = imputer.fit_transform(x_test)
original_dim = arg.input_dim
input_shape = (original_dim,)
latent_dim = 2

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(256, activation='relu')(inputs)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)

z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(64, activation='relu')(latent_inputs)
x = Dense(128, activation='relu')(x)
x = Dense(256, activation='relu')(x)

outputs = Dense(original_dim, dtype = tf.float32)(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

os.makedirs(arg.model_dir, exist_ok=True)
outputfile_latent = open('%s/%s'%(arg.model_dir,arg.output_latent),"w")
vae.summary()
plot_model(vae,to_file='vae_model.png',show_shapes=True)

if __name__ == '__main__':
    data = (imputed_x_test, y_test)
    models = (encoder, decoder)
    vae.compile()
    vae.load_weights('%s/%s'%(arg.model_dir,arg.input_weights))
    write_results(models,
                 data,
                 model_name=arg.model_dir)

