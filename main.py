import numpy as np
from read_images import read_images
from model import autoencoder_model
from model import train_model
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import CosineSimilarity
from pathlib import Path

if __name__ == '__main__':
    
    train_directory = 'data/train'
    test_directory = 'data/test'
    # read_images(train_directory)
    # read_images(test_directory, test = True)
    X = np.load("X.npy")
    # train_model(X, epochs = 1, bs = 64, validation_split = 0.2)
    # model = load_model('trained_model.h5')
    # encoded_data = model.predict(X)
    # np.save('image_encodings.npy', encoded_data)
    
    encodings_train = np.load('image_encodings.npy')
    encodings_test = np.load('X_test.npy')
    dir = Path(train_directory)
    imgs=dir.glob('*jpg')
    names = [str(el) for el in imgs]
    similarities = []
    cosine_sim = CosineSimilarity(axis = 1)
    for i in range(encodings_train.shape[0]):
        sim = cosine_sim(encodings_test[0], encodings_train[i]).numpy()
        print(sim)
        similarities.append(sim)
    f = dict(zip(names, similarities))
    
    
    f = sorted(f.items(), key=lambda x: x[1])[:10]
    print(f)
        
    
    
    # SIMILARITY
    
    
    
    
    
    