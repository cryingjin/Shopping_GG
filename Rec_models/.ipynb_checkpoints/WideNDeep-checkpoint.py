import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder, MinMaxScaler
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, SpatialDropout1D, Activation, concatenate, Conv2D, Conv1D, Lambda, GlobalMaxPooling1D
from keras.optimizers import Adam, SGD
from keras.layers.advanced_activations import ReLU, PReLU, LeakyReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from tensorflow.keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy, top_k_categorical_accuracy

def Deep_model(data, COLUMNS, CATEGORICAL_COLUMNS, CONTINUOUS_COLUMNS):
   
    category_inputs = []
    category_embeds = []
    
    # categorical columns embedding
    for i in range(len(CATEGORICAL_COLUMNS)):
        input_i = Input(shape=(1, ), dtype = 'int32')
        dim = len(np.unique(data[CATEGORICAL_COLUMNS[i]]))
        embed_dim = int(np.ceil(dim ** 0.5))
        embed_i = Embedding(dim, embed_dim, input_length = 1)(input_i)
        flatten_i = Flatten()(embed_i)
        category_inputs.append(input_i)
        category_embeds.append(flatten_i)
        
    # continuous columns input
    continue_input = Input(shape = (len(CONTINUOUS_COLUMNS), ))
    continue_dense = Dense(256, use_bias = False)(continue_input)
    
    # categorical & continue CONCAT
    concat_embeds = concatenate([continue_dense] + category_embeds)
    concat_embeds = Activation('relu')(concat_embeds)
    bn_concat = BatchNormalization()(concat_embeds)

    fc1 = Dense(512, use_bias=False)(bn_concat)
    relu1 = ReLU()(fc1)
    bn1 = BatchNormalization()(relu1)
    
    fc2 = Dense(256, use_bias=False)(bn1)
    relu2 = ReLU()(fc2)
    bn2 = BatchNormalization()(relu2)
    
    fc3 = Dense(128)(bn2)
    relu3 = ReLU()(fc3)

    return category_inputs, continue_input, relu3

def Wide_model(X_CATEGORICAL_COLUMNS_POLY):
    n = X_CATEGORICAL_COLUMNS_POLY.shape[1]
    return Input(shape = (n,))

def top_3_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)