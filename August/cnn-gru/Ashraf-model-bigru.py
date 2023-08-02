import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix

####EDIT BEFORE RUNNING ###########
NUM_CLASSES = 10

# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "../../Stage1/audio_file/preprocessed/310genre_dataset.json"
SAVE_MODEL = True
SAVE_HM = True

#OUTPUT DIR/FILE NAMES
NEWDIR_PATH = "genre"

MODEL_NAME = "saved_model"
HM_NAME = "heatmap.png"
A_PLOT_NAME = 'accuracy.png'
L_PLOT_NAME = 'loss.png'

# Hyperparameters
LEARNING_RATE = 0.0001
EPOCHS = 50

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

def create_combined_model(cnn_input_shape, rnn_input_shape, num_classes):
    cnn_input = keras.Input(shape=cnn_input_shape)
    rnn_input = keras.Input(shape=rnn_input_shape)

    cnn_model = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(cnn_input)
    cnn_model = keras.layers.Dropout(0.25)(cnn_model)
    cnn_model = keras.layers.MaxPooling1D(pool_size=2)(cnn_model)
    
    cnn_model = keras.layers.Conv1D(64, 5, activation='relu', padding='same')(cnn_model)
    cnn_model = keras.layers.Dropout(0.25)(cnn_model)
    cnn_model = keras.layers.MaxPooling1D(pool_size=2)(cnn_model)
    
    cnn_model = keras.layers.Conv1D(128, 7, activation='relu', padding='same')(cnn_model)
    cnn_model = keras.layers.Dropout(0.25)(cnn_model)
    cnn_model = keras.layers.MaxPooling1D(pool_size=2)(cnn_model)
    
    cnn_model = keras.layers.Conv1D(256, 9, activation='relu', padding='same')(cnn_model)
    cnn_model = keras.layers.Dropout(0.25)(cnn_model)
    cnn_model = keras.layers.MaxPooling1D(pool_size=2)(cnn_model)
    
    cnn_model = keras.layers.Conv1D(512, 11, activation='relu', padding='same')(cnn_model)
    cnn_model = keras.layers.Dropout(0.25)(cnn_model)
    cnn_model = keras.layers.MaxPooling1D(pool_size=2)(cnn_model)

    cnn_model = keras.layers.Flatten()(cnn_model)

    rnn_model = keras.layers.GRU(128, return_sequences=True)(rnn_input)
    rnn_model = keras.layers.Dropout(0.25)(rnn_model)
    rnn_model = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True))(rnn_model)
    rnn_model = keras.layers.Dropout(0.25)(rnn_model)
    rnn_model = keras.layers.GRU(64)(rnn_model)
    rnn_model = keras.layers.Dropout(0.25)(rnn_model)

    rnn_model = keras.layers.Flatten()(rnn_model)
    
    combined = keras.layers.concatenate([cnn_model, rnn_model])
    output = keras.layers.Dense(num_classes, activation='softmax')(combined)

    model = keras.Model(inputs=[cnn_input, rnn_input], outputs=output)
    return model
    

if __name__ == "__main__":
    # create train, val, test sets
    cnn_X_train, cnn_X_validation, cnn_X_test, cnn_y_train, cnn_y_validation, cnn_y_test, cnn_label_list = prepare_cnn_datasets(0.25, 0.2)
    rnn_X_train, rnn_X_validation, rnn_X_test, rnn_y_train, rnn_y_validation, rnn_y_test, rnn_label_list = prepare_rnn_datasets(0.25, 0.2)

    # Define the input shapes and number of classes
    cnn_input_shape = (cnn_X_train.shape[1], cnn_X_train.shape[2])  # Assumes input audio features of shape (num_timesteps, num_features)
    rnn_input_shape = (rnn_X_train.shape[1], rnn_X_train.shape[2])
    num_classes = NUM_CLASSES  # Number of music genres

    # Create the combined model
    model = create_combined_model(cnn_input_shape, rnn_input_shape, num_classes)
    
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
    
    # Print validation loss and accuracy
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print("Valdiation Loss: ", val_loss)
    print("Valdiation Accuracy: ", val_acc)

    # Plot history
    save_plot(history)

    # Evaluate model on test set
    test_loss, test_acc = model.evaluate(cnn_X_test, cnn_y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    # Pick a sample to predict from the test set
    X_to_predict = cnn_X_test[10]
    y_to_predict = cnn_y_test[10]

    # Predict sample
    #predict(model, X_to_predict, y_to_predict)

    # Save model
    if SAVE_MODEL:
        model.save(os.path.join(NEWDIR_PATH, MODEL_NAME))
        print("Model saved to disk at:", os.path.join(NEWDIR_PATH, MODEL_NAME))

    # Output heatmap
    if SAVE_HM:
        get_heatmap(model, cnn_X_test, cnn_y_test, NEWDIR_PATH, HM_NAME, cnn_label_list)
    
