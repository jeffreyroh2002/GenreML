import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

####EDIT BEFORE RUNNING ###########
# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "../../audio_file/preprocessed/full_dataset0510.json"
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

# Define the CNN model
def create_cnn(input_shape):
    cnn_model = keras.Sequential()

    cnn_model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    cnn_model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    cnn_model.add(keras.layers.BatchNormalization())

    cnn_model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    cnn_model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    cnn_model.add(keras.layers.BatchNormalization())

    cnn_model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    cnn_model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    cnn_model.add(keras.layers.BatchNormalization())

    cnn_model.add(keras.layers.Flatten())
    return cnn_model

# Define the bi-directional RNN model
def create_rnn(input_shape):
    # build network topology
    rnn_model = keras.Sequential()

    # 2 LSTM layers
    rnn_model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    rnn_model.add(keras.layers.LSTM(64))

    # dense layer
    rnn_model.add(keras.layers.Dense(64, activation='relu'))
    rnn_model.add(keras.layers.Dropout(0.3))

    # output layer
    rnn_model.add(keras.layers.Dense(10, activation='softmax'))
    return rnn_model


# Define the combined model
def create_combined_model(cnn_input_shape, rnn_input_shape, num_classes):
    cnn_model = create_cnn(cnn_input_shape)
    rnn_model = create_rnn(rnn_input_shape)
    
    merged = keras.layers.Concatenate()([cnn_model.output,rnn_model.output])

    z = keras.layers.Dense(128, activation="relu")(merged)
    z = keras.layers.Dense(num_classes, activation='softmax')
    
    combined_model = keras.Model(inputs=[cnn_model.input,rnn_model.input], outputs=z)
    
    combined_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    """
    combined_model.add(keras.layers.concatenate([cnn_model.output, rnn_model.output]))
    combined_model.add(keras.layers.Dense(128, activation='relu'))
    combined_model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    combined_model = keras.Sequential()
    """

    return combined_model


def predict(model, X, y):
    """Predict a single sample using the trained model

    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """
    # add a dimension to input data for sample - model.predict() expects a 4d array in this case

    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))

    
if __name__ == "__main__":
    # create train, val, test sets
    cnn_X_train, cnn_X_validation, cnn_X_test, cnn_y_train, cnn_y_validation, cnn_y_test, cnn_label_list = prepare_cnn_datasets(0.25, 0.2)
    rnn_X_train, rnn_X_validation, rnn_X_test, rnn_y_train, rnn_y_validation, rnn_y_test, rnn_label_list = prepare_rnn_datasets(0.25, 0.2)

    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Define the input shapes and number of classes
    cnn_input_shape = (cnn_X_train.shape[1], cnn_X_train.shape[2], 1) # Assumes input audiofeatures of shape (num_timesteps, num_features)
    rnn_input_shape = (rnn_X_train.shape[1], rnn_X_train.shape[2])
    
    num_classes = 9  # Number of music genres
    
    print("0")
    
    # Create the combined model
    model = create_combined_model(cnn_input_shape, rnn_input_shape, num_classes)
    
    print("1")
    
    optimiser = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Compile the model
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("2")
    # Print the model summary
    model.summary()

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=EPOCHS,
                        verbose=1) 
    
    print("Finished Training Model!")