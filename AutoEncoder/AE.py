import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


#(82x1133)
#57

class AutoEncoder(tf.keras.Model):
    def __init__(self, X_size, latent_dim):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.X_size = X_size
        self.encoder = tf.keras.Sequential([
            layers.Dense(512, activation = 'selu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(256, activation = 'selu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation = 'selu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(latent_dim, activation = 'selu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(128,activation = 'selu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(256,activation = 'selu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(512,activation = 'selu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(X_size,activation = 'sigmoid')
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def loss(x, x_bar):
    return tf.losses.mean_squared_error(x, x_bar)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        reconstruction = model(inputs)
        loss_value = loss(targets, reconstruction)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), reconstruction

if __name__ == '__main__':
    data = pd.read_excel("Rec_user_item_matrix2.xlsx")
    X = data.iloc[:,1:]

    x_train, x_test, _, _ = train_test_split(np.asarray(X),np.asarray(X),test_size=0.1,shuffle=False,random_state=1004)


    model = AutoEncoder(x_train.shape[1], 64)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    global_step = tf.Variable(0)

    num_epochs = 1000
    batch_size = 4

    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        for x in range(0, len(x_train), batch_size):
            x_inp = x_train[x : x + batch_size]
            loss_value, grads, reconstruction = grad(model, x_inp, x_inp)
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                  global_step)
            
            if global_step.numpy() % 500 == 0:
                print("Step: {},         Loss: {}".format(global_step.numpy(),
                                              loss(x_inp, reconstruction).numpy()))

