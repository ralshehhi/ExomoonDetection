import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.backend import conv1d, conv2d
from keras.models import Model
from keras.layers import Convolution1D, MaxPooling1D, Dropout, GaussianDropout, LeakyReLU
from keras.layers import Input, Dense, Flatten 
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.optimizers import RMSprop

from functools import partial
from scipy import signal

window = signal.gaussian(9,std=0.3)
kwindow = K.variable(window)
kwindow1 = K.expand_dims(kwindow,)
kwindow2 = K.expand_dims(kwindow1,)

def get_net(input_shape):

	inputs = Input(shape=input_shape)

	DROPOUT_RATE=0.2 
	INNERLAYER_DROPOUT_RATE=0.3

	x = BatchNormalization(axis=1, mode=0, name='bn_0_freq')(inputs)

	x = Convolution1D(64, 3, padding="same", name='conv1')(x)
	x = BatchNormalization(axis=1, name='bn1')(x)
	x = LeakyReLU(alpha=0.02)(x)
	x = MaxPooling1D(2, name='pool1')(x)
	x = GaussianDropout(INNERLAYER_DROPOUT_RATE, name='dropout1')(x)

	x = Convolution1D(128, 3, padding="same", name='conv2')(x)
	x = BatchNormalization(axis=1, name='bn2')(x)
	x = LeakyReLU(alpha=0.02)(x)
	x = MaxPooling1D(2, name='pool2')(x)
	x = GaussianDropout(INNERLAYER_DROPOUT_RATE, name='dropout2')(x)

	x = Convolution1D(128, 3, padding="same", name='conv3')(x)
	x = BatchNormalization(axis=1, name='bn3')(x)
	x = LeakyReLU(alpha=0.02)(x)
	x = MaxPooling1D(2, name='pool3')(x)
	x = GaussianDropout(INNERLAYER_DROPOUT_RATE, name='dropout3')(x)

	x = Convolution1D(128, 3, padding="same", name='conv4')(x)
	x = BatchNormalization(axis=1, name='bn4')(x)
	x = LeakyReLU(alpha=0.02)(x)
	x = MaxPooling1D(3, name='pool4')(x)
	x = GaussianDropout(INNERLAYER_DROPOUT_RATE, name='dropout4')(x)

	x = Convolution1D(64, 3, padding="same", name='conv5')(x)
	x = BatchNormalization(axis=1, name='bn5')(x)
	x = LeakyReLU(alpha=0.02)(x)
	x = MaxPooling1D(4, name='pool5')(x)
	x = GaussianDropout(INNERLAYER_DROPOUT_RATE, name='dropout5')(x)

	x = Flatten()(x)
	x=Dense(2048, activation='relu', name='fc1')(x)
	x=Dropout(0.5)(x)
	x = Dense(1, activation='sigmoid', name='output')(x)
	
	model = Model(inputs, x)
	print(model.summary())
	
	return model

def get_loss(klayer, w=1):
	def loss(y_true,y_pred):
		l1 = K.mean((y_pred - y_true), axis=-1)
		klayer1 =  K.expand_dims(klayer,2)
		kconv = conv1d(klayer1, kwindow2, padding='same')
		a = K.abs(kconv[:,kconv.shape[1]-1] - kconv[:,1:])
		l2 = K.mean(a,axis=1)
		return (l1+l2)
	return loss
