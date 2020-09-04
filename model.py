from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, UpSampling2D, Activation
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
def autoencoder_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(224,224,3)))
    model.add(MaxPool2D((2,2), padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D((2,2), padding='same'))
    model.add(Dropout(0.2))


    model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
    model.add(UpSampling2D((2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(UpSampling2D((2,2)))
    model.add(Dropout(0.2))


    model.add(Conv2D(3, kernel_size=3, padding='same', activation='relu'))

    model.compile(optimizer='adam', loss="mse")
    model.summary()
    model.save('model.h5')
    print(" Model saved in model.h5 ...")
    print(" Model Craeted ...")
    return model

def plot_train_val_loss():
    pass

def train_model(X, epochs = 50, bs = 64, validation_split = 0.2):
    model = autoencoder_model()
    model.fit(X, X, epochs=epochs, batch_size=bs, shuffle=True, validation_split=validation_split)
    model.save('trained_model.h5')
    model = load_model('trained_model.h5')
    encoded_data = model.predict(X)
    np.save('image_encodings.npy', encoded_data)
    print("Image Encodings saved in image_encodings.npy ...")
    print("Model Training Complete ...")


