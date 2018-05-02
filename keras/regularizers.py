import keras.backend as K

def zeromean_regularizer(weight_matrix=None, amt=0.01):
    return amt * K.abs(K.mean(weight_matrix))

def unitymean_regularizer(weight_matrix=None, amt=0.2):
    return amt * K.abs(1.0-K.mean(weight_matrix))

import keras.regularizers
keras.regularizers.zeromean_regularizer = zeromean_regularizer
keras.regularizers.unitymean_regularizer = unitymean_regularizer
