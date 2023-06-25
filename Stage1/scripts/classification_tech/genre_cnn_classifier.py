import os
import sys
import cv2
import seaborn as sn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.utils.multiclass import unique_labels
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, LSTM, LeakyReLU, Input
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, MaxPool1D, GaussianNoise, GlobalMaxPooling1D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
import tensorflow
import matplotlib.pyplot as plt
import datetime

# OUTPUT DIR/FILE NAMES
NEWDIR_NAME = "genre_cnn-" + str(datetime.date.today())
DATA_PATH = "/workspace/MusicML/Stage1/audio_file/preprocessed/GTZAN_features"
MODEL_PATH = "/workspace/MusicML/Stage1/results/" + NEWDIR_NAME

# create new dr in results dir for results
NEWDIR_PATH = os.path.join("../../results", NEWDIR_NAME)
os.makedirs(NEWDIR_PATH)


def save_confusion_matrix(model, input_train, input_test,  label_train, label_test, file_path):
    label_pred = model.predict(input_train)
    label_pred = np.argmax(label_pred, axis=-1)
    label_true = np.argmax(label_train, axis=-1)

    correct = len(label_pred) - np.count_nonzero(label_pred - label_true)
    acc = correct / len(label_pred)
    acc = np.round(acc, 4) * 100

    print("Train Accuracy: ", correct, "/", len(label_pred), " = ", acc, "%")

    # Testing Accuracy
    label_pred = model.predict(input_test)
    label_pred = np.argmax(label_pred, axis=-1)
    label_true = np.argmax(label_test, axis=-1)

    correct = len(label_pred) - np.count_nonzero(label_pred - label_true)
    acc = correct / len(label_pred)
    acc = np.round(acc, 4) * 100

    print("Test Accuracy: ", correct, "/", len(label_pred), " = ", acc, "%")

    class_names = ["Blues", "Classical", "Country",
                   "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae"]
    conf_mat = confusion_matrix(label_true, label_pred, normalize='true')
    conf_mat = np.round(conf_mat, 2)

    conf_mat_df = pd.DataFrame(
        conf_mat, columns=class_names, index=class_names)

    plt.figure(figsize=(10, 7), dpi=200)
    sn.set(font_scale=1.4)
    sn.heatmap(conf_mat_df, annot=True, annot_kws={"size": 16})  # font size
    plt.tight_layout()
    plt.savefig(file_path)
    print("Succesfully saved heatmap at: ", file_path)

# def save_plot(history, newdir_path=MODEL_PATH, a_plot_name="accuracy.png", l_plot_name="loss.png"):

#     #Outputing graphs for Accuracy
#     plt.figure()
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('model train_accuracy vs val_accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train','val'], loc='upper left')
#     plt.savefig(os.path.join(newdir_path, a_plot_name))

#     #Outputing graphs for Loss
#     plt.figure()
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model train_loss vs val_loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train','val'], loc='upper left')
#     plt.savefig(os.path.join(newdir_path, l_plot_name))


def spec_cnn_model1(input_shape):

    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation='relu',
              input_shape=input_shape, padding='same'))
    model.add(MaxPooling2D((4, 4), padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((4, 4), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((4, 4), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((4, 4), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((4, 4), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def spec_cnn_model2(input_shape):
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation='relu',
              input_shape=input_shape, padding='same'))
    model.add(MaxPooling2D((4, 4), padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((4, 4), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((4, 4), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((4, 4), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((4, 4), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def mfcc_cnn_model1(input_shape):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=input_shape,
              activation='tanh', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((4, 6), padding='same'))
    model.add(Conv2D(32, (3, 3), input_shape=input_shape,
              activation='tanh', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((4, 6), padding='same'))
    model.add(Conv2D(64, (3, 3), input_shape=input_shape,
              activation='tanh', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((4, 6), padding='same'))
    model.add(Flatten())
    # model.add(Dense(256, activation= 'tanh'))
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def mfcc_cnn_model2(input_shape):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=input_shape,
              activation='tanh', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((4, 6), padding='same'))
    model.add(Conv2D(32, (3, 3), input_shape=input_shape,
              activation='tanh', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((4, 6), padding='same'))
    model.add(Conv2D(64, (3, 3), input_shape=input_shape,
              activation='tanh', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((4, 6), padding='same'))
    model.add(Flatten())
    # model.add(Dense(256, activation= 'tanh'))
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def mfcc_cnn_model3(input_shape):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=input_shape,
              activation='tanh', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((4, 6), padding='same'))
    model.add(Conv2D(32, (3, 3), input_shape=input_shape,
              activation='tanh', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((4, 6), padding='same'))
    model.add(Conv2D(64, (3, 3), input_shape=input_shape,
              activation='tanh', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((4, 6), padding='same'))
    model.add(Flatten())
    # model.add(Dense(256, activation= 'tanh'))
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def mel_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation='relu',
              input_shape=input_shape, padding='same'))
    model.add(MaxPooling2D((4, 4), padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((4, 4), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((4, 4), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((4, 4), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((4, 4), padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def majority_vote(pred):
    vote = []

    for i in range(len(pred[0])):
        candidates = [x[i] for x in pred]
        candidates = np.array(candidates)
        uniq, freq = np.unique(candidates, return_counts=True)
        vote.append(uniq[np.argmax(freq)])

    vote = np.array(vote)

    return vote


def mfcc_result_matrix(mfcc_train, mfcc_test, y_train, y_test):
    # Load the models
    model1 = load_model(MODEL_PATH + "/saved_model/model_mfcc_1.h5")
    model2 = load_model(MODEL_PATH + "/saved_model/model_mfcc_2.h5")
    model3 = load_model(MODEL_PATH + "/saved_model/model_mfcc_3.h5")

    # Training Accuracy
    y_true = np.argmax(y_train, axis=-1)

    y_pred1 = model1.predict(mfcc_train)
    y_pred1 = np.argmax(y_pred1, axis=-1)

    y_pred2 = model2.predict(mfcc_train)
    y_pred2 = np.argmax(y_pred2, axis=-1)

    y_pred3 = model3.predict(mfcc_train)
    y_pred3 = np.argmax(y_pred3, axis=-1)

    y_pred = [y_pred1, y_pred2, y_pred3]

    y_pred = majority_vote(y_pred)

    correct = len(y_pred) - np.count_nonzero(y_pred - y_true)
    acc = correct / len(y_pred)
    acc = np.round(acc, 4) * 100

    print("Train Accuracy: ", correct, "/", len(y_pred), " = ", acc, "%")

    # Test Model
    y_true = np.argmax(y_test, axis=-1)

    y_pred1 = model1.predict(mfcc_test)
    y_pred1 = np.argmax(y_pred1, axis=-1)

    y_pred2 = model2.predict(mfcc_test)
    y_pred2 = np.argmax(y_pred2, axis=-1)

    y_pred3 = model3.predict(mfcc_test)
    y_pred3 = np.argmax(y_pred3, axis=-1)

    y_pred = [y_pred1, y_pred2, y_pred3]

    y_pred = majority_vote(y_pred)

    correct = len(y_pred) - np.count_nonzero(y_pred - y_true)
    acc = correct / len(y_pred)
    acc = np.round(acc, 4) * 100

    print("Testing Accuracy: ", correct, "/", len(y_pred), " = ", acc, "%")

    class_names = ["Blues", "Classical", "Country",
                   "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae"]
    conf_mat = confusion_matrix(y_true, y_pred, normalize='true')
    conf_mat = np.round(conf_mat, 2)

    conf_mat_df = pd.DataFrame(
        conf_mat, columns=class_names, index=class_names)

    plt.figure(figsize=(10, 7), dpi=200)
    sn.set(font_scale=1.4)
    sn.heatmap(conf_mat_df, annot=True, annot_kws={"size": 16})  # font size
    plt.tight_layout()
    plt.savefig(MODEL_PATH + "/mfcc_ensemble_heatmap.png")
    print("Succesfully saved heatmap at: ",
          MODEL_PATH + "/mfcc_ensemble_heatmap.png")


def ensemble_result_matrix(S_train, S_test, mfcc_train, mfcc_test, mel_train, mel_test, y_train, y_test):
    model1 = load_model(MODEL_PATH+"/saved_model/model_spec_1.h5")
    model2 = load_model(MODEL_PATH+"/saved_model/model_spec_2.h5")
    model3 = load_model(MODEL_PATH+"/saved_model/model_mfcc_1.h5")
    model4 = load_model(MODEL_PATH+"/saved_model/model_mfcc_2.h5")
    model5 = load_model(MODEL_PATH+"/saved_model/model_mfcc_3.h5")
    model6 = load_model(MODEL_PATH+"/saved_model/model_melspectrogram.h5")

    # Ground truth
    y_true = np.argmax(y_train, axis=-1)
    # Train model
    # Spec model 1
    y_pred1 = model1.predict(S_train)
    y_pred1 = np.argmax(y_pred1, axis=-1)

    # Spec model 2
    y_pred2 = model2.predict(S_train)
    y_pred2 = np.argmax(y_pred2, axis=-1)

    # MFCC model 1
    y_pred3 = model3.predict(mfcc_train)
    y_pred3 = np.argmax(y_pred3, axis=-1)

    # MFCC model 2
    y_pred4 = model4.predict(mfcc_train)
    y_pred4 = np.argmax(y_pred4, axis=-1)

    # MFCC model 3
    y_pred5 = model5.predict(mfcc_train)
    y_pred5 = np.argmax(y_pred5, axis=-1)

    # Mel-spectrogram
    y_pred6 = model6.predict(mel_train)
    y_pred6 = np.argmax(y_pred6, axis=-1)

    # majority vote
    y_pred = [y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6]
    y_pred = majority_vote(y_pred)

    correct = len(y_pred) - np.count_nonzero(y_pred - y_true)
    acc = correct / len(y_pred)
    acc = np.round(acc, 4) * 100

    print("Training Accuracy: ", correct, "/", len(y_pred), " = ", acc, "%")

    # Test Model
    y_true = np.argmax(y_test, axis=-1)
    # Spectrogram model 1
    y_pred1 = model1.predict(S_test)
    y_pred1 = np.argmax(y_pred1, axis=-1)

    # Spectrogram model 2
    y_pred2 = model2.predict(S_test)
    y_pred2 = np.argmax(y_pred2, axis=-1)

    # MFCC model 1
    y_pred3 = model3.predict(mfcc_test)
    y_pred3 = np.argmax(y_pred3, axis=-1)

    # MFCC model 2
    y_pred4 = model4.predict(mfcc_test)
    y_pred4 = np.argmax(y_pred4, axis=-1)

    # MFCC model 3
    y_pred5 = model5.predict(mfcc_test)
    y_pred5 = np.argmax(y_pred5, axis=-1)

    # Mel-Spectrogram
    y_pred6 = model6.predict(mel_test)
    y_pred6 = np.argmax(y_pred6, axis=-1)

    # Get majority vote
    y_pred = [y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6]
    y_pred = majority_vote(y_pred)

    correct = len(y_pred) - np.count_nonzero(y_pred - y_true)
    acc = correct / len(y_pred)
    acc = np.round(acc, 4) * 100
    print("Testing Accuracy: ", correct, "/", len(y_pred), " = ", acc, "%")

    class_names = ["Blues", "Classical", "Country",
                   "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae"]
    conf_mat = confusion_matrix(y_true, y_pred, normalize='true')
    conf_mat = np.round(conf_mat, 2)

    conf_mat_df = pd.DataFrame(
        conf_mat, columns=class_names, index=class_names)

    plt.figure(figsize=(10, 7), dpi=200)
    sn.set(font_scale=1.4)
    sn.heatmap(conf_mat_df, annot=True, annot_kws={"size": 16})  # font size
    plt.tight_layout()
    plt.savefig(MODEL_PATH + "/new_ensemble_heatmap.png")
    print("Succesfully saved heatmap for model ensemble!")


if __name__ == "__main__":
    # Edit befor running
    SAVE_SPEC_1 = True
    SAVE_SPEC_2 = True
    SAVE_MFCC_1 = True
    SAVE_MFCC_2 = True
    SAVE_MFCC_3 = True
    SAVE_MEL = True
    # predict the genre based on those 6 models
    ENSEMBLE = True

    # Model for Spectrogram
    spec_file = np.load(DATA_PATH + "/new_spectrogram_train_test.npz")

    S_train = spec_file['S_train']
    S_test = spec_file['S_test']
    S_y_train = spec_file['y_train']
    S_y_test = spec_file['y_test']

    # Model for MFCC
    mfcc_file = np.load(DATA_PATH+"/new_mfcc_train_test.npz")

    mfcc_train = mfcc_file['mfcc_train']
    mfcc_test = mfcc_file['mfcc_test']
    MF_y_train = mfcc_file['y_train']
    MF_y_test = mfcc_file['y_test']

    # Model for Mel-Spectrogram
    mel_file = np.load(DATA_PATH+"/new_mel_train_test.npz")
    mel_train = mel_file['mel_train']
    mel_test = mel_file['mel_test']
    mel_y_train = mel_file['y_train']
    mel_y_test = mel_file['y_test']

    # SPEC_1 MODEL
    if SAVE_SPEC_1 == True:
        spec_model_1 = spec_cnn_model1(S_train[0].shape)

        spec_model_1.fit(S_train, S_y_train, epochs=100,
                         batch_size=32, verbose=1)
        print("Successfully Trained SPEC_1 Model")

        spec_model_1.save(MODEL_PATH+"/saved_model/model_spec_1.h5")
        print("Model saved to disk at: ", MODEL_PATH +
              "/saved_model/model_spec_1.h5")
        save_confusion_matrix(spec_model_1, S_train, S_test,
                              S_y_train, S_y_test, MODEL_PATH+"/heatmap_spec_1.png")

    # SPEC_2 MODEL
    if SAVE_SPEC_2 == True:
        spec_model_2 = spec_cnn_model2(S_train[0].shape)

        spec_model_2.fit(S_train, S_y_train, epochs=100,
                         batch_size=32, verbose=1)
        print("Successfully Trained SPEC_2 Model")

        spec_model_2.save(MODEL_PATH+"/saved_model/model_spec_2.h5")
        print("Model saved to disk at: ", MODEL_PATH +
              "/saved_model/model_spec_2.h5")
        save_confusion_matrix(spec_model_2, S_train, S_test,
                              S_y_train, S_y_test, MODEL_PATH+"/heatmap_spec_2.png")

    # MFCC_1 MODEL
    if SAVE_MFCC_1 == True:
        mfcc_model_1 = mfcc_cnn_model1(mfcc_train[0].shape)

        kf = KFold(n_splits=10)
        for train_index, val_index in kf.split(mfcc_train, np.argmax(MF_y_train, axis=-1)):

            kf_mfcc_train = mfcc_train[train_index]
            kf_X_val = mfcc_train[val_index]
            kf_y_train = MF_y_train[train_index]
            kf_y_val = MF_y_train[val_index]

            mfcc_model_1.fit(kf_mfcc_train, kf_y_train, validation_data=(
                kf_X_val, kf_y_val), epochs=30, batch_size=30, verbose=1)
            print("Successfully Trained MFCC_1 Model")

            mfcc_model_1.save(MODEL_PATH+"/saved_model/model_mfcc_1.h5")
            print("Model saved to disk at: ", MODEL_PATH +
                  "/saved_model/model_mfcc_1.h5")

    # MFCC_2 MODEL
    if SAVE_MFCC_2 == True:
        mfcc_model_2 = mfcc_cnn_model2(mfcc_train[0].shape)

        kf = KFold(n_splits=10)
        for train_index, val_index in kf.split(mfcc_train, np.argmax(MF_y_train, axis=-1)):

            kf_mfcc_train = mfcc_train[train_index]
            kf_X_val = mfcc_train[val_index]
            kf_y_train = MF_y_train[train_index]
            kf_y_val = MF_y_train[val_index]

            mfcc_model_2.fit(kf_mfcc_train, kf_y_train, validation_data=(
                kf_X_val, kf_y_val), epochs=30, batch_size=30, verbose=1)
            print("Successfully Trained MFCC_2 Model")

            mfcc_model_2.save(MODEL_PATH+"/saved_model/model_mfcc_2.h5")
            print("Model saved to disk at: ", MODEL_PATH +
                  "/saved_model/model_mfcc_2.h5")

    # MFCC_3 MODEL
    if SAVE_MFCC_3 == True:
        mfcc_model_3 = mfcc_cnn_model3(mfcc_train[0].shape)

        kf = KFold(n_splits=10)
        for train_index, val_index in kf.split(mfcc_train, np.argmax(MF_y_train, axis=-1)):

            kf_mfcc_train = mfcc_train[train_index]
            kf_X_val = mfcc_train[val_index]
            kf_y_train = MF_y_train[train_index]
            kf_y_val = MF_y_train[val_index]

            mfcc_model_3.fit(kf_mfcc_train, kf_y_train, validation_data=(
                kf_X_val, kf_y_val), epochs=30, batch_size=30, verbose=1)
            print("Successfully Trained MFCC_3 Model")

            mfcc_model_3.save(MODEL_PATH+"/saved_model/model_mfcc_3.h5")
            print("Model saved to disk at: ", MODEL_PATH +
                  "/saved_model/model_mfcc_3.h5")

    if SAVE_MFCC_1 and SAVE_MFCC_2 and SAVE_MFCC_3:
        mfcc_result_matrix(mfcc_train, mfcc_test, MF_y_train, MF_y_test)

    if SAVE_MEL == True:
        mel_model = mel_cnn_model(mel_train[0].shape)

        mel_model.fit(mel_train, mel_y_train, epochs=150,
                      batch_size=32, verbose=1)
        mel_model.save(MODEL_PATH + "/saved_model/model_melspectrogram.h5")
        print("Successfully saved model at : ", MODEL_PATH +
              "/saved_model/model_melspectrogram.h5")
        save_confusion_matrix(mel_model, mel_train, mel_test,
                              mel_y_train, mel_y_test, MODEL_PATH+"/heatmap_mel.png")

    if ENSEMBLE == True:
        ensemble_result_matrix(S_train, S_test, mfcc_train,
                               mfcc_test, mel_train, mel_test, mel_y_train, mel_y_test)
