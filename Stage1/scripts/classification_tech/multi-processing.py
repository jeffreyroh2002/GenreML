import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import multiprocessing

####EDIT BEFORE RUNNING ###########
# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "../../audio_file/preprocessed/310genre_dataset.json"

# Hyperparameters
LEARNING_RATE = 0.0001
EPOCHS = 50

####################################
def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    label_list = data.get("mapping", {})

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
    X_train = X_train[..., np.newaxis]
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

def create_combined_model(cnn_input_shape, rnn_input_shape, num_classes, hidden_size):
    cnn_input = keras.Input(shape=cnn_input_shape)
    rnn_input = keras.Input(shape=rnn_input_shape)

    cnn_model = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(cnn_input)
    cnn_model = keras.layers.ReLU()(cnn_model)
    cnn_model = keras.layers.MaxPooling1D(pool_size=2, strides=2)(cnn_model)
    cnn_model = keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same')(cnn_model)
    cnn_model = keras.layers.ReLU()(cnn_model)
    cnn_model = keras.layers.MaxPooling1D(pool_size=2, strides=2)(cnn_model)
    cnn_model = keras.layers.Flatten()(cnn_model)
    cnn_model = keras.layers.Dense(128)(cnn_model)
    cnn_model = keras.layers.ReLU()(cnn_model)

    rnn_model = keras.layers.Bidirectional(keras.layers.GRU(units=hidden_size, return_sequences=True))(rnn_input)
    rnn_model = keras.layers.Bidirectional(keras.layers.GRU(units=hidden_size))(rnn_model)
    rnn_model = keras.layers.Dense(128, activation='relu')(rnn_model)
    rnn_model = keras.layers.Dropout(0.3)(rnn_model)

    combined = keras.layers.concatenate([cnn_model, rnn_model])
    output = keras.layers.Dense(num_classes, activation='softmax')(combined)

    model = keras.Model(inputs=[cnn_input, rnn_input], outputs=output)
    return model

def train_cnn_model(cnn_model, cnn_X_train, cnn_X_validation, cnn_y_train, cnn_y_validation):
    optimiser = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    cnn_model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(cnn_X_train, cnn_y_train, validation_data=(cnn_X_validation, cnn_y_validation),
                  batch_size=32, epochs=EPOCHS, verbose=1)

def train_rnn_model(rnn_model, rnn_X_train, rnn_X_validation, rnn_y_train, rnn_y_validation):
    optimiser = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    rnn_model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'])
    rnn_model.fit(rnn_X_train, rnn_y_train, validation_data=(rnn_X_validation, rnn_y_validation),
                  batch_size=32, epochs=EPOCHS, verbose=1)

if __name__ == "__main__":
    # create train, val, test sets
    cnn_X_train, cnn_X_validation, cnn_X_test, cnn_y_train, cnn_y_validation, cnn_y_test, cnn_label_list = prepare_cnn_datasets(0.25, 0.2)
    rnn_X_train, rnn_X_validation, rnn_X_test, rnn_y_train, rnn_y_validation, rnn_y_test, rnn_label_list = prepare_rnn_datasets(0.25, 0.2)

    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Define the input shapes and number of classes
    cnn_input_shape = (cnn_X_train.shape[1], cnn_X_train.shape[2])
    rnn_input_shape = (rnn_X_train.shape[1], rnn_X_train.shape[2])
    num_classes = 10

    print("CNN input shape:", cnn_input_shape)
    print("RNN input shape:", rnn_input_shape)
    print("X_train shape:", cnn_X_train.shape)

    # Define the hidden size for GRU layers
    hidden_size = 64

    # Create the CNN and RNN models
    cnn_model = create_combined_model(cnn_input_shape, rnn_input_shape, num_classes, hidden_size)
    rnn_model = create_combined_model(cnn_input_shape, rnn_input_shape, num_classes, hidden_size)

    # Create separate processes for trainingthe CNN and RNN models
    cnn_process = multiprocessing.Process(target=train_cnn_model, args=(cnn_model, cnn_X_train, cnn_X_validation, cnn_y_train, cnn_y_validation))
    rnn_process = multiprocessing.Process(target=train_rnn_model, args=(rnn_model, rnn_X_train, rnn_X_validation, rnn_y_train, rnn_y_validation))

    # Start the processes
    cnn_process.start()
    rnn_process.start()

    # Wait for the processes to finish
    cnn_process.join()
    rnn_process.join()

    print("Finished Training Models!")
