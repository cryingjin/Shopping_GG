import os
import sys
import json
import joblib
import time
import argparse
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

import WideNDeep as WnD


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required = True)
parser.add_argument('--epoch', required = True, type=int)
parser.add_argument('--lr', required = True, type=float)
parser.add_argument('--batch', required = True, type=int)
args = parser.parse_args()

epoch = args.epoch
lr = args.lr
batch = args.batch

data = joblib.load(os.path.join( '..', 'data', '05_분석데이터', args.dataset))

locals().update(data)

COLUMNS = list(X.columns)
CATEGORICAL_COLUMNS = ['isHoliday', '평일여부', '방송시간대', '계절', '분기']
CONTINUOUS_COLUMNS = list(set(COLUMNS) - set(CATEGORICAL_COLUMNS))


x_train_continue, x_train_category, x_train_category_poly, train_label = data4train[0], data4train[1], data4train[2], data4train[3]
x_test_continue, x_test_category, x_test_category_poly, test_label = data4valid[0], data4valid[1], data4valid[2], data4valid[3]

category_inputs, continue_input, deep_model = WnD.Deep_model(X, COLUMNS, CATEGORICAL_COLUMNS, CONTINUOUS_COLUMNS)
wide_model = WnD.Wide_model(x_train_category_poly)

out_layer = concatenate([deep_model, wide_model])
inputs = [continue_input] + category_inputs + [wide_model]
output = Dense(len(train_label[0]), activation='softmax')(out_layer)
model = Model(inputs=inputs, outputs=output)


early_stopping = EarlyStopping(monitor = 'val_top_3_categorical_accuracy', patience = 30)

input_data = [x_train_continue] + [x_train_category[:, i] for i in range(x_train_category.shape[1])] + [x_train_category_poly]


def top_3_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k = 3)

optimizer = Adam(learning_rate = lr)

model.compile(optimizer = optimizer,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy',  top_3_categorical_accuracy]
             )

model.fit(input_data,
          train_label,
          epochs = epoch,
          batch_size = batch,
          validation_split = 0.2,
          callbacks = [early_stopping]
         )

model.save('./model/model_'+ time.strftime('%Y-%m-%d', time.localtime(time.time())) +'.h5')
model_json = model.to_json()

with open('./model/model_'+ time.strftime('%Y-%m-%d', time.localtime(time.time())) +'.json', 'w') as f:
    json.dump(model_json, f)
model.save_weights('./model/model_weights_'+ time.strftime('%Y-%m-%d', time.localtime(time.time())) +'.h5')

print('model saved!')        