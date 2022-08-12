''' VAE model traning for MD data set
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
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.python.keras.layers import Lambda, Input, Dense, Flatten, Embedding, Dropout, Attention
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

import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir',type=str, default='model',
    help='Base directory for the model.')

parser.add_argument(
    '--train_epochs', type=int, default=10, help='Number of training epochs.')

parser.add_argument(
    '--batch_size', type=int, default=4, help="Number of batches.")

parser.add_argument(
    '--sample_number', type=int, default=135, help="Number of samples.")

parser.add_argument(
    '--input_dim', type=int, default=62, help="Dimension of input parameter tensor.")

parser.add_argument(
    '--optimizer', type=str, default='Adam', help='Optimizer that is either Adam or Adagrad.')

parser.add_argument(
    '--learning_rate', type=float, default=0.01, help='Learning rate.')

parser.add_argument(
    '--trainfile', type=str, default='allprop.csv', help='Datafile for the training set.')

parser.add_argument(
    '--testfile', type=str, default='allprop.csv', help='Datafile for the test set.')

parser.add_argument(
    '--output_latent', type=str, default='latent.dat', help='Write the latent space variable into a text file.')

parser.add_argument(
    '--output_loss', type=str, default='loss.dat', help='Write the loss into a text file.')

parser.add_argument(
    '--output_weights', type=str, default='weights.h5', help='Write the weights.')

arg = parser.parse_args()

def prepare_dist_data(trainname):

    label_encoder = LabelEncoder()

    train_data = genfromtxt(trainname, delimiter=',')
    train_features = train_data[:,range(train_data[0].size-2)]
    train_label = train_data[:,range(train_data[0].size-2,train_data[0].size-1)]
    train_datamut = genfromtxt(trainname, delimiter=',', dtype='unicode')
    train_mutants = train_datamut[:,range(train_datamut[0].size-1,train_datamut[0].size)]
    return train_features,train_label,train_mutants

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
                 batch_size=10,
                 model_name="vae"):

    encoder, decoder = models
    x_test, y_test = data

    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    z0 = []
    z1 = []
    for i in range(samplenumber):
      z0.append((z_mean[:, 0][i]-(min(z_mean[:, 0])+(max(z_mean[:, 0])-min(z_mean[:, 0]))/2))*10/(max(z_mean[:, 0])-min(z_mean[:, 0])))
      z1.append((z_mean[:, 1][i]-(min(z_mean[:, 1])+(max(z_mean[:, 1])-min(z_mean[:, 1]))/2))*10/(max(z_mean[:, 1])-min(z_mean[:, 1])))
    for s in range(samplenumber):
      print(x_mutants[s][0],'z0',z0[s],'z1',z1[s],'label',y_test[s][0],file=outputfile_latent)

    outputfile_latent.close()
x_train,y_train,x_mutants = prepare_dist_data(arg.trainfile)
samplenumber=arg.sample_number
original_dim = arg.input_dim
input_shape = (original_dim,)
batch_size = arg.batch_size
epochs = arg.train_epochs
latent_dim = 2

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(128, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Attention()([x,x])
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='%s/vae_att_encoder.png'%arg.model_dir, show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(32, activation='relu')(latent_inputs)
x = Attention()([x,x])
x = Dense(62, activation='relu')(x)
x = Dense(128, activation='relu')(x)

outputs = Dense(original_dim, dtype = tf.float32)(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='%s/vae_att_decoder.png'%arg.model_dir, show_shapes=True)
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

os.makedirs(arg.model_dir, exist_ok=True)
outputfile_latent = open('%s/%s'%(arg.model_dir,arg.output_latent),"w")
vae.summary()
plot_model(vae,to_file='%s/vae_att_model.png'%arg.model_dir,show_shapes=True)

if __name__ == '__main__':
    data = (x_train, y_train)
    models = (encoder, decoder)
    reconstruction_loss = mse(inputs, outputs[0])
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean((reconstruction_loss + kl_loss)/1.0)
    vae.add_loss(vae_loss)
    if arg.optimizer=="Adagrad":
      opt = Adagrad(lr=arg.learning_rate)
    else:
      opt = Adam(lr=arg.learning_rate)
    vae.compile(optimizer=opt)
    vae.summary()

    modeltl = vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_train, None))
    vae.save_weights('%s/%s'%(arg.model_dir,arg.output_weights))
    loss_history = modeltl.history["loss"]
    np_loss_history = np.array(loss_history)
    np.savetxt('%s/%s'%(arg.model_dir,arg.output_loss), np_loss_history, delimiter=",", fmt='%f')
    write_results(models,
                 data,
                 batch_size=batch_size,
                 model_name=arg.model_dir)


