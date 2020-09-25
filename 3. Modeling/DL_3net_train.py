
from __future__ import division
import joblib
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras import layers,optimizers,metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Conv1D
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM,Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from ops import *
from utils import *




def DL_model(X_num,X_emb):

    def create_mlp(dim):
        # define our MLP network
        model = Sequential()
        model.add(Dense(64, input_dim=dim, activation ='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(32, activation ='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(10, activation ='relu', kernel_initializer='he_normal'))   
        return model

    def create_1Dcnn(dim):
        inputShape = (dim,1)

        Inputs = Input(shape = inputShape)

        conv1 = Conv1D(filters = 16, kernel_size=6,padding = 'valid',activation ='linear', kernel_initializer='he_normal')(Inputs)
        pool1 = GlobalAveragePooling1D()(conv1)

        conv2 = Conv1D(filters = 16, kernel_size=7,padding = 'valid', activation ='linear', kernel_initializer='he_normal')(Inputs)
        pool2 = GlobalAveragePooling1D()(conv2)

        conv3 = Conv1D(filters = 16, kernel_size=8,padding = 'valid', activation ='linear', kernel_initializer='he_normal')(Inputs)
        pool3 = GlobalAveragePooling1D()(conv3)

        concat = concatenate([pool1, pool2, pool3])
        #concat = tf.expand_dims(concat,-1)

        #results = LSTM(64)(concat)
        results = Dense(10,activation ='linear', kernel_initializer='he_normal')(concat)
        model = Model(Inputs,results)
        
        return model


    def create_lstm(dim):
        inputShape = (dim,1)
        
        inputs = Input(shape = inputShape)
        print(inputs.shape)
        
        x = LSTM(20, return_sequences=True, kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = LSTM(10, kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(10,activation ='relu', kernel_initializer='he_normal')(x)
        model = Model(inputs,x)

        return model


    mlp = create_mlp(X_num.shape[1])
    cnn = create_1Dcnn(X_emb.shape[1])
    lstm = create_lstm((X_time.shape[1]))
    #print(mlp.output, lstm.output)

    combinedInput = concatenate([mlp.output, cnn.output,lstm.output])
    #combinedInput = tf.expand_dims(combinedInput,-1)

    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    #x = LSTM(5)(combinedInput)
    x = Dense(32, activation="selu")(combinedInput)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(8, activation="selu")(combinedInput)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(1, activation="selu")(x)

    model = Model(inputs=[mlp.input, cnn.input, lstm.input], outputs=x)
    return model



def DataLoad_DL(data_dir,timeseries_list_dir):

    data = joblib.load(data_dir)
    locals().update(data)

    
    
    X = data['X']
    y = data['y'] 

    
    
    column_list =[]
    emb_list = ['v'+str(j) for j in range(0,110)]
    with open(timeseries_list_dir, 'rb') as f:
        time_ser = pkl.load(f)
    
    for col in X.columns:
        column_list.append(col)
    for i in emb_list:
        column_list.remove(i)

    num_list = column_list

    X_num = X[num_list]
    X_emb = X[emb_list]
    X_time = X_tmp[time_ser]
    X_time = X_time.fillna(0)
    X_num = X_num.fillna(0)


    return X, X_num, X_emb, X_time, y


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./train_FE.pkl')
    parser.add_argument('--model_dir', type=str, default='./model_3multiNet.h5',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--timeS_dir', type=str, default='./timeseries_list.pkl',
                        help='Directory name to save the checkpoints')
                        
    return check_args(parser.parse_args())

def main():

    preds = {'val_preds' : [], 'test_preds' : []} 
    mape = {'val_mape' : [], 'test_mape' : []} 

    args = ap.parse_args()
    if args is None:
      exit()

    X, X_num, X_emb, y = DataLoad_DL(arg.data_dir,timeS_dir)
    model = DL_model(X_num,X_emb)

    opt = Adam(lr=0.0001, decay=1e-3 / 200)

    model.compile(loss= "mean_absolute_percentage_error",optimizer=opt)
    
    reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=30)
    earlystopping = EarlyStopping(monitor='loss',patience= 100)
    
    for i in range(1,13):
        print('처리중인 월:',i)

        train_idx = X[X['방송월'] != i ].index
        test_idx = X[X['방송월'] == i ].index

        X_train_num = X_num.loc[train_idx]
        X_train_emb = X_emb.loc[train_idx]
        X_train_time = X_time.loc[train_idx]
        y_train = y.loc[X_train_num.index]

        X_train_emb = np.asarray(X_train_emb).astype(np.float32)
        X_train_emb = np.reshape(X_train_emb,(X_train_emb.shape[0],X_train_emb.shape[1],1))
        X_train_num = np.asarray(X_train_num).astype(np.float32)
        X_train_num = np.reshape(X_train_num,(X_train_num.shape[0],X_train_num.shape[1],1))
        X_train_time = np.asarray(X_train_time).astype(np.float32)
        X_train_time = np.reshape(X_train_time,(X_train_time.shape[0],X_train_time.shape[1],1))
        y_train = np.asarray(y_train).astype(np.float32)


        test_num = X_num.loc[test_idx]
        test_emb = X_emb.loc[test_idx]
        test_time = X_time.loc[test_idx]

        X_val_num = test_num.loc[((X['방송일'] > 0) & (X['방송일'] <15))]
        X_val_emb = test_emb.loc[X_val_num.index]
        X_val_time = test_time.loc[X_val_num.index]
        y_val = y.loc[X_val_num.index]

        X_test_num = test_num.loc[((X['방송일'] > 16) & (X['방송일'] < 32))]
        X_test_emb = test_emb.loc[X_test_num.index]
        X_test_time = test_time.loc[X_test_num.index]
        y_test = y.loc[X_test_num.index]



        X_test_emb = np.asarray(X_test_emb).astype(np.float32)
        X_test_emb = np.reshape(X_test_emb,(X_test_emb.shape[0],X_test_emb.shape[1],1))
        X_test_num = np.asarray(X_test_num).astype(np.float32)
        X_test_num = np.reshape(X_test_num,(X_test_num.shape[0],X_test_num.shape[1],1))
        X_test_time = np.asarray(X_test_time).astype(np.float32)
        X_test_time = np.reshape(X_test_time,(X_test_time.shape[0],X_test_time.shape[1],1))

        y_test = np.asarray(y_test).astype(np.float32)
        
        
        X_val_emb = np.asarray(X_val_emb).astype(np.float32)
        X_val_emb = np.reshape(X_val_emb,(X_val_emb.shape[0],X_val_emb.shape[1],1))
        X_val_num = np.asarray(X_val_num).astype(np.float32)
        X_val_num = np.reshape(X_val_num,(X_val_num.shape[0],X_val_num.shape[1],1))
        X_val_time = np.asarray(X_val_time).astype(np.float32)
        X_val_time = np.reshape(X_val_time,(X_val_time.shape[0],X_val_time.shape[1],1))

        y_val = np.asarray(y_val).astype(np.float32)
    
        print(X_train_num.shape, X_train_emb.shape,X_train_time.shape, y_train.shape)
        print(X_val_num.shape, X_val_emb.shape, X_val_time.shape, y_val.shape)
        print(X_test_num.shape, X_test_emb.shape, X_test_time.shape, y_test.shape)

        
        model.fit(
        x=[X_train_num, X_train_emb, X_train_time], y=y_train,
        validation_data=([X_val_num, X_val_emb,X_val_time], y_val),
        epochs=2000, batch_size = 256,
        callbacks = [reduceLR,earlystopping])

        y_pred = model.predict([X_test_num, X_test_emb,X_test_time])
        val_pred = model.predict([X_val_num, X_val_emb, X_val_time])
        """
        preds['val_preds'].append(val_pred)
        preds['test_preds'].append(y_pred)
        mape['val_mape'].append(mean_absolute_percentage_error(y_val, val_pred))
        mape['test_mape'].append(mean_absolute_percentage_error(y_test,y_pred))
        """

        preds['val_preds'].append(np.exp(val_pred))
        preds['test_preds'].append(np.exp(y_pred))
        mape['val_mape'].append(mean_absolute_percentage_error(np.exp(y_val), np.exp(val_pred)))
        mape['test_mape'].append(mean_absolute_percentage_error(np.exp(y_test), np.exp(y_pred)))

        for m, arg in enumerate(zip(mape['val_mape'], mape['test_mape']), 1):
                print(f'{m}월\t', '[val]:', arg[0], '\t[test]', arg[1]) 


        model.save(arg.model_dir) 




if __name__ == '__main__':
    main()
 