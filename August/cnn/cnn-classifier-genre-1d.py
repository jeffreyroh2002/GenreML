import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import tensorflow.keras as keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

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
EPOCHS = 100

#create new dr in results dir for results
if not os.path.exists(NEWDIR_PATH):
    os.makedirs(NEWDIR_PATH)

####################################

def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    label_list = data.get("mapping", {})  # ['Anger', 'Fear', ...]

    print(label_list)
    print("Data succesfully loaded!")

    return  X, y, label_list

def save_plot(history, newdir_path=NEWDIR_PATH, a_plot_name=A_PLOT_NAME, l_plot_name=L_PLOT_NAME):

        # Outputting graphs for Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model train_accuracy vs val_accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(newdir_path, a_plot_name))
    plt.close()

    # Outputting graphs for Loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train_loss vs val_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(newdir_path, l_plot_name))
    plt.close()

    
def get_heatmap(model, X_test, y_test, newdir_path, hm_name, label_list):
    
    plt.figure()
    
    #extracting predictions of X_test
    prediction = model.predict(X_test)
    y_pred = np.argmax(prediction, axis=1)
    #cm = confusion_matrix(y_test, y_pred)

    labels = sorted(label_list)
    column = [f'Predicted {label}' for label in labels]
    indices = [f'Actual {label}' for label in labels]
    table = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=column, index=indices)
    plt.figure()
    
    hm = sns.heatmap(table, annot=True, fmt='d', cmap='viridis')
    plt.savefig(os.path.join(newdir_path, hm_name))
    plt.close()
    print("heatmap generated and saved in {path}".format(path=newdir_path))

def prepare_datasets(test_size, validation_size):
    
    # load data
    X, y, label_list = load_data(DATA_PATH)
    
    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    
    # add an axis to input sets (CNN requires 3D array)
    # X_train = X_train[..., np.newaxis]    #4d array -> (num_samples, 130, 13, 1
    # X_validation = X_validation[..., np.newaxis]
    # X_test = X_test[..., np.newaxis]
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test, label_list
    
def build_model(input_shape):
    """Generates CNN model

    :param input_shape (tuple): Shape of input set
    :return model: CNN model
    """

    # build network topology
    input = keras.layers.Input(shape=input_shape)
    
    cnn_model = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(input)
    # cnn_model = keras.layers.BatchNormalization()(cnn_model)
    cnn_model = keras.layers.Dropout(0.25)(cnn_model)
    cnn_model = keras.layers.MaxPooling1D(pool_size=2)(cnn_model)
    cnn_model = keras.layers.Conv1D(64, 5, activation='relu', padding='same')(cnn_model)
    # cnn_model = keras.layers.BatchNormalization()(cnn_model)
    cnn_model = keras.layers.Dropout(0.25)(cnn_model)
    cnn_model = keras.layers.MaxPooling1D(pool_size=2)(cnn_model)
    cnn_model = keras.layers.Conv1D(128, 7, activation='relu', padding='same')(cnn_model)
    # cnn_model = keras.layers.BatchNormalization()(cnn_model)
    cnn_model = keras.layers.Dropout(0.25)(cnn_model)
    cnn_model = keras.layers.MaxPooling1D(pool_size=2)(cnn_model)
    cnn_model = keras.layers.Conv1D(256, 9, activation='relu', padding='same')(cnn_model)
    # cnn_model = keras.layers.BatchNormalization()(cnn_model)
    cnn_model = keras.layers.Dropout(0.25)(cnn_model)
    cnn_model = keras.layers.MaxPooling1D(pool_size=2)(cnn_model)
    cnn_model = keras.layers.Conv1D(512, 11, activation='relu', padding='same')(cnn_model)
    # cnn_model = keras.layers.BatchNormalization()(cnn_model)
    cnn_model = keras.layers.Dropout(0.25)(cnn_model)
    cnn_model = keras.layers.MaxPooling1D(pool_size=2)(cnn_model)
    cnn_model = keras.layers.Conv1D(256, 11, activation='relu', padding='same')(cnn_model)
    cnn_model = keras.layers.Dropout(0.25)(cnn_model)
    cnn_model = keras.layers.MaxPooling1D(pool_size=2)(cnn_model)
    cnn_model = keras.layers.Flatten()(cnn_model)
    
    output = keras.layers.Dense(10, activation='softmax')(cnn_model)

    model = keras.Model(inputs=[input], outputs=[output])
    
    return model

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

    #create train, val, test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test, label_list = prepare_datasets(0.25, 0.2)

    # create networ
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)
    
    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=EPOCHS, verbose=1)
    print("Finished Training Model!")
    
    #printing val loss and accuracy
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print("Valdiation Loss: ", val_loss)
    print("Valdiation Accuracy: ", val_acc)
    
    #plot history
    save_plot(history)
    
    #evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
    
    # pick a sample to predict from the test set
    X_to_predict = X_test[10]
    y_to_predict = y_test[10]

    # predict sample
    predict(model, X_to_predict, y_to_predict)
    
    # save model
    if (SAVE_MODEL == True):
        model.save(os.path.join(NEWDIR_PATH, MODEL_NAME))
        print("Model saved to disk at: ", os.path.join(NEWDIR_PATH, MODEL_NAME))

    # output heatmap
    if (SAVE_HM == True):
        get_heatmap(model, X_test, y_test, NEWDIR_PATH, HM_NAME, label_list)
