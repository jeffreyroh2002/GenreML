import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import tensorflow.keras as keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import pickle
import os

# path to json file that stores MFCCs and genre labels for each processed segment
DATA_PATH = "../../audio_file/preprocessed/full_dataset0510.json"
SAVE_MODEL = True
SAVE_HM = True
NEWDIR_NAME = "0510-testing"

#create new directory in results if model or hm is saved
if SAVE_MODEL or SAVE_HM:
    NEWDIR_PATH = os.path.join("../../results", NEWDIR_NAME)
    #os.makedirs(NEWDIR_PATH, exist_ok=True)

MODEL_NAME = "saved_model"
HM_NAME = "heatmap.png"

#PICKLE_PATH = "{newdir_path}{model_name}".format(newdir_path=NEWDIR_PATH, model_name=MODEL_NAME)
#HM_PATH = "{newdir_path}{hm_name}".format(newdir_path=NEWDIR_PATH, hm_name=HM_NAME)

def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return  X, y




if __name__ == "__main__":

    # load data
    X, y = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # build network topology
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        # 1st dense layer
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 2nd dense layer
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(10, activation='softmax')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=100, verbose=0)
	
    if (SAVE_MODEL == True):
        model.save(os.path.join(NEWDIR_PATH, MODEL_NAME))
        print("Model saved to disk at: ", os.path.join(NEWDIR_PATH, MODEL_NAME))

    if (SAVE_HM == True):
        #extracting predictions of X_test
        prediction = model.predict(X_test)
        y_pred = np.argmax(prediction, axis=1)
        cm = confusion_matrix(y_test, y_pred)
        #print(cm)

        labels = unique_labels(y_test)
        column = [f'Predicted {label}' for label in labels]
        indices = [f'Actual {label}' for label in labels]
        table = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=column, index=indices)
        hm = sns.heatmap(table, annot=True, fmt='d', cmap='viridis')
        plt.savefig(os.path.join(NEWDIR_PATH, HM_NAME))
        #print(hm)
        print("heatmap generated and saved in {path}".format(path=NEWDIR_PATH))