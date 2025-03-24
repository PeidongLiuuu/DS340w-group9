# Enhanced Human Activity Recognition using LSTM for Time-Series Data

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from imblearn.over_sampling import SMOTE
from Extrasensory_Manipulation import *
from Inputs_HDLAct import *

warnings.filterwarnings("ignore")
os.chdir("HHAR-Net-master\\HHAR-net")


# Data cleaning and preprocessing function
def data_cleaner(dataset, feature_set_range, labels):
    valid_labels = [label for label in labels if label in dataset.columns and dataset[label].notna().sum() > 0]
    dataset[valid_labels] = dataset[valid_labels].fillna(0)
    data = pd.concat([dataset.iloc[:, feature_set_range], dataset[valid_labels]], axis=1).dropna()

    if data.empty:
        return None, None

    scaler = StandardScaler()
    X = scaler.fit_transform(data.iloc[:, :len(feature_set_range)])

    y = data[valid_labels].idxmax(axis=1)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return X, y


# LSTM-based hybrid model definition
def create_lstm_model(input_shape, output_dim, dropout_rate=0.3):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(LSTM(64))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Main workflow
if __name__ == "__main__":
    # Load dataset
    dataset_uuids = readdata_csv(data_dir)
    dataset = pd.concat(dataset_uuids.values(), axis=0)

    sensors_list = sensors()
    feature_set_range = [feature for sensor in sensors_to_use for feature in sensors_list[sensor]]

    labels = ['label:OR_standing', 'label:SITTING', 'label:LYING_DOWN',
              'label:FIX_running', 'label:FIX_walking', 'label:BICYCLING']

    # Clean and preprocess data
    X, y = data_cleaner(dataset, feature_set_range, labels)
    if X is not None and y is not None:
        # Balance the dataset
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        # Prepare train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

        # Reshape data for LSTM (samples, timesteps, features)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Compute class weights for balanced training
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))

        # Define and train LSTM model
        model = create_lstm_model(input_shape=(X_train.shape[1], 1), output_dim=len(np.unique(y_train)))

        checkpoint = ModelCheckpoint('best_lstm_model.keras', save_best_only=True, verbose=1)

        model.fit(X_train, y_train, batch_size=32, epochs=30, validation_split=0.2,
                  class_weight=class_weight_dict, callbacks=[checkpoint])

        # Evaluate the model
        y_pred = np.argmax(model.predict(X_test), axis=1)

        # Results
        print("F1 Score (Macro):", f1_score(y_test, y_pred, average='macro'))
        print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
        print("Overall Accuracy:", accuracy_score(y_test, y_pred))
