# -*- coding: utf-8 -*-
import os
import warnings
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
from imblearn.over_sampling import SMOTE
from Extrasensory_Manipulation import *
from Inputs_HDLAct import *

# Ignore warnings
warnings.filterwarnings("ignore")

# Set working directory
os.chdir("C:\\Users\\BridgeSword\\Desktop\\HHAR-Net-master\\HHAR-net")

# Function to clean and prepare data
def data_cleaner(dataset, feature_set_range, parent_labels):
    print("Dataset before cleaning:")
    print(dataset.head())

    # Validate parent labels
    valid_parent_labels = [label for label in parent_labels if label in dataset.columns and dataset[label].notna().sum() > 0]
    print(f"Valid parent labels: {valid_parent_labels}")

    # Fill missing labels with 0
    dataset[valid_parent_labels] = dataset[valid_parent_labels].fillna(0)

    # Extract features and labels
    features = dataset.iloc[:, feature_set_range]
    labels = dataset[valid_parent_labels]
    raw_data = pd.concat([features, labels], axis=1).dropna()

    if raw_data.empty:
        print("No valid samples remaining after cleaning.")
        return None, None

    # Normalize features
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(raw_data.iloc[:, :len(feature_set_range)])

    # Encode labels
    y = raw_data[valid_parent_labels].idxmax(axis=1)
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y

# Function to define improved deep learning model
def create_optimized_model(input_dim, output_dim, dropout_rate=0.3):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(input_dim, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(64)))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Main function
if __name__ == "__main__":
    f1_accuracy = {}
    BA_accuracy = {}
    accuracy = {}

    # Load and preprocess data
    dataset_uuids = readdata_csv(data_dir)
    uuids = list(dataset_uuids.keys())
    dataset = pd.concat([dataset_uuids[uuid] for uuid in uuids], axis=0)

    sensors_list = sensors()
    feature_set_range = [feature for sensor in sensors_to_use for feature in sensors_list[sensor]]
    print(f"Feature set range: {feature_set_range}")

    parent_labels = ['label:OR_standing', 'label:SITTING', 'label:LYING_DOWN', 'label:FIX_running', 'label:FIX_walking', 'label:BICYCLING']
    X, y = data_cleaner(dataset, feature_set_range, parent_labels)

    if X is not None and y is not None:
        # Handle class imbalance using SMOTE
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X, y = smote.fit_resample(X, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Compute class weights
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))

        # Reshape input for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Define and train model
        model = create_optimized_model(input_dim=X_train.shape[1], output_dim=len(np.unique(y_train)))
        best_model_path = "./optimized_model.keras"
        checkpointer = ModelCheckpoint(filepath=best_model_path, verbose=1, save_best_only=True)

        model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2, class_weight=class_weight_dict, callbacks=[checkpointer])

        # Evaluate model
        y_pred = np.argmax(model.predict(X_test), axis=1)
        f1_accuracy['parent'] = f1_score(y_test, y_pred, average='macro')
        BA_accuracy['parent'] = balanced_accuracy_score(y_test, y_pred)
        accuracy['parent'] = accuracy_score(y_test, y_pred)

        print("F1 Accuracy:", f1_accuracy)
        print("Balanced Accuracy:", BA_accuracy)
        print("Accuracy:", accuracy)
