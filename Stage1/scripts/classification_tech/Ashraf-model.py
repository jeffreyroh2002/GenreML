import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers


####EDIT BEFORE RUNNING ###########
# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "../../audio_file/preprocessed/310genre_dataset.json"
SAVE_MODEL = True
SAVE_HM = True

#OUTPUT DIR/FILE NAMES
NEWDIR_NAME = "genre_bi-dir-rnn-cnn-0706-100epochs"

MODEL_NAME = "saved_model"
HM_NAME = "heatmap.png"
A_PLOT_NAME = 'accuracy.png'
L_PLOT_NAME = 'loss.png'

# Hyperparameters
LEARNING_RATE = 0.0001
EPOCHS = 100

####################################

#create new dir in results dir for results
NEWDIR_PATH = os.path.join("../../results", NEWDIR_NAME)
if not os.path.exists(NEWDIR_PATH):
    os.makedirs(NEWDIR_PATH)

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    label_list = data.get("mapping", {})   #Jazz, Classical, etc

    print(label_list)

    print("Data successfully loaded!")

    return X, y, label_list

def prepare_cnn_datasets(test_size, validation_size):
    # load data
    X, y, label_list = load_data(DATA_PATH)

    # create train, validation, and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    
    # add an axis to input sets (CNN requires 3D array)
    X_train = X_train[..., np.newaxis]    #4d array -> (num_samples, 130, 13, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test, label_list
    
def prepare_rnn_datasets(test_size, validation_size):

    # load data
    X, y, label_list = load_data(DATA_PATH)

    # create train, validation, and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test, label_list

###########
# Convolution block
def conv_block(inputs, filters, kernel_size):
    x = layers.Conv1D(filters, kernel_size, activation='relu', padding='same')(inputs)
    x = layers.Dropout(0.25)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    return x

def create_cnn(input_shape):
    model = keras.Sequential()

    # Add convolutional blocks
    model.add(conv_block(layers.Input(shape=input_shape), filters=32, kernel_size=3))
    model.add(conv_block(model.layers[-1], filters=64, kernel_size=5))
    model.add(conv_block(model.layers[-1], filters=128, kernel_size=7))
    model.add(conv_block(model.layers[-1], filters=256, kernel_size=9))
    model.add(conv_block(model.layers[-1], filters=512, kernel_size=11))

    # Flatten the output
    model.add(layers.Flatten())
    
    return model

def create_rnn(input_shape):
    model = keras.Sequential()
    
    # Add LSTM layers
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Dropout(0.25))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Dropout(0.25))
    model.add(layers.LSTM(64))
    model.add(layers.Dropout(0.25))

    # Flatten LSTM output
    model.add(layers.Flatten())

#######
if __name__ == "__main__":
        
    # create train, val, test sets
    cnn_X_train, cnn_X_validation, cnn_X_test, cnn_y_train, cnn_y_validation, cnn_y_test, cnn_label_list = prepare_cnn_datasets(0.25, 0.2)
    rnn_X_train, rnn_X_validation, rnn_X_test, rnn_y_train, rnn_y_validation, rnn_y_test, rnn_label_list = prepare_rnn_datasets(0.25, 0.2)
    
    # Define the input shapes and number of classes
    cnn_input_shape = (cnn_X_train.shape[1], cnn_X_train.shape[2])  # Assumes input audio features of shape (num_timesteps, num_features)
    rnn_input_shape = (rnn_X_train.shape[1], rnn_X_train.shape[2])
    num_classes = 10  # Number of music genres
    
    cnn_input = keras.Input(shape=cnn_input_shape)
    rnn_input = keras.Input(shape=rnn_input_shape)
    
    combined = keras.layers.concatenate([create_cnn, create_rnn])
    combined.add(layers.Dense(128, activation='relu'))
    output = keras.layers.Dense(num_classes, activation='softmax')(combined)
    
    model = keras.Model(inputs=[cnn_input, rnn_input], outputs=output)

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    optimiser = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Compile the model
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Print the model summary
    model.summary()

    # Train the model
    history = model.fit([cnn_X_train, rnn_X_train], cnn_y_train, validation_data=([cnn_X_validation, rnn_X_validation], cnn_y_validation),
                        batch_size=32, epochs=EPOCHS, verbose=1)
    
    print("Finished Training Model!")
