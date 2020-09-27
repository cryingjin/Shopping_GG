import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import argparse
import joblib
import os
from sklearn.model_selection import train_test_split


#(82x1133)
#57
import tensorflow as tf
from tensorflow.keras import layers, losses
import numpy as np

#(82x1133)
#57

class AutoEncoder(tf.keras.Model):
    def __init__(self, X_size, latent_dim):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.X_size = X_size
        self.encoder = tf.keras.Sequential([
            layers.Dense(latent_dim, activation = 'selu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2)
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(X_size,activation = 'selu')
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def masked_mse(y_true, y_pred):
    # masked function
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    # masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_mse = K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
    return masked_mse

def loss(y_true, y_pred): #masked rmse
        # masked function
        mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
        # masked squared error
        masked_squared_error = K.square(mask_true * (y_true - y_pred))
        masked_mse = K.sqrt(K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1))
        return masked_mse

def masked_rmse_clip(y_true, y_pred):
    # masked function
    mask_true = K.cast(K.not_equal(y_true, 0), K.floatx())
    y_pred = K.clip(y_pred, 1, 10)
    # masked squared error
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_mse = K.sqrt(K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1))
    return masked_mse

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        reconstruction = model(inputs)
        loss_value = loss(targets, reconstruction)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), reconstruction

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.path.join( '..', 'data', '04_임시데이터', 'user_item_matrix.pkl'),
                        help='Directory name to load the user item matrix')
    parser.add_argument('--data_type', type=str, default='log',
                        help='log or origin')
    args = parser.parse_args()
    data_dir = args.data_dir
    data_type = args.data_type

    data = joblib.load(data_dir)
    locals().update(data)
    logdf = (logdf + 1).fillna(0)
    df = (df + 1).fillna(0)
    if data_type == 'log':
        X = logdf
    else : 
        X = df

    x_train, x_test, _, _ = train_test_split(np.asarray(X),np.asarray(X),test_size=0.1,shuffle=False,random_state=1004)


    model = AutoEncoder(x_train.shape[1], 64)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    global_step = tf.Variable(0)

    num_epochs = 500
    batch_size = 4

    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        for x in range(0, len(x_train), batch_size):
            x_inp = x_train[x : x + batch_size]
            loss_value, grads, reconstruction = grad(model, x_inp, x_inp)
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                  global_step)
            
        if epoch%100 == 0:
            print("Step: {},         Loss: {}".format(global_step.numpy(),
                                            loss(x_inp, reconstruction).numpy()))

    encoded_num = model.encoder(np.asarray(X)).numpy()
    decoded_num = model.decoder(encoded_num).numpy()
    
    X_en = pd.DataFrame(decoded_num)

    X_en.to_excel("X_encoded.xlsx")
