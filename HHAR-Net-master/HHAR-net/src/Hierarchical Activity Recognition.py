# -*- coding: utf-8 -*-
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Attention, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from imblearn.over_sampling import SMOTE
from Extrasensory_Manipulation import *
from Inputs_HDLAct import *

warnings.filterwarnings("ignore")
os.chdir("..\\..\\")


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


# Enhanced LSTM with Attention Mechanism
def create_attention_bi_lstm_model(input_shape, output_dim, dropout_rate=0.3):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = BatchNormalization()(x)

    # Attention Mechanism
    attention = Attention()([x, x])
    attention_output = Dense(64, activation='relu')(attention)
    attention_output = Dropout(dropout_rate)(attention_output)

    outputs = Dense(output_dim, activation='softmax')(attention_output[:, -1, :])
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Main workflow
if __name__ == "__main__":
    # Load dataset
    dataset_uuids = readdata_csv(data_dir)
    dataset = pd.concat(list(dataset_uuids.values()), axis=0)

    sensors_list = sensors()
    feature_set_range = [feature for sensor in sensors_to_use for feature in sensors_list[sensor]]

    labels = [
        'label:OR_standing',
        'label:SITTING',
        'label:LYING_DOWN',
        'label:FIX_running',
        'label:FIX_walking',
        'label:BICYCLING'
    ]

    # Data cleaning and preprocessing
    X, y = data_cleaner(dataset, feature_set_range, labels)
    if X is not None and y is not None:
        # Handle imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42
        )

        # Reshape data for LSTM (samples, timesteps, features)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Compute class weights for balanced training
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))

        # Define and train Enhanced Bidirectional LSTM model
        model = create_attention_bi_lstm_model(
            input_shape=(X_train.shape[1], 1),
            output_dim=len(np.unique(y_train))
        )

        checkpoint = ModelCheckpoint('best_attention_bi_lstm_model.keras', save_best_only=True, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model with EarlyStopping
        model.fit(
            X_train, y_train,
            batch_size=32, epochs=50, validation_split=0.2,
            class_weight=class_weight_dict, callbacks=[checkpoint, early_stopping]
        )

        # Evaluate the model
        y_pred = np.argmax(model.predict(X_test), axis=1)

        # Results
        print("\nEnhanced Attention-BiLSTM Evaluation Metrics:")
        print("F1 Score (Macro):", f1_score(y_test, y_pred, average='macro'))
        print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
        print("Overall Accuracy:", accuracy_score(y_test, y_pred))
