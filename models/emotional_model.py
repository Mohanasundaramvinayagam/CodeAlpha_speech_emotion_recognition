import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

def train_and_save_model():
    print(" Loading extracted features...")

# Load the features from the pickle file
    with open("features/mfcc_features.pkl", "rb") as f:
        df = pickle.load(f)

    if not isinstance(df, pd.DataFrame):
        raise ValueError(" Feature data is not a DataFrame.")

    print(" Feature data loaded")
    print(f" DataFrame columns: {df.columns}")
    print(f" First few rows:\n{df.head()}")

    
    X_raw = df['features'].tolist()
    y_raw = df['label'].tolist()

    if len(X_raw) == 0:
        raise ValueError(" No feature data found. Check feature extraction step.")

    
    X = np.array(X_raw)
    y = np.array(y_raw)

    print(f" Shape of X before reshape: {X.shape}")
    print(f" Sample labels: {np.unique(y)}")

# Label encoding
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    
    os.makedirs("models", exist_ok=True)
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
        print(" Saved label encoder to models/label_encoder.pkl")


    y_cat = to_categorical(y_encoded)

# Reshape X for LSTM
    X = X.reshape((X.shape[0], 1, X.shape[1]))

# Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y_cat.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(" Training model...")
    model.fit(X, y_cat, epochs=30, batch_size=16, verbose=1)

    
    model.save("models/emotion_lstm_model.h5")
    print(" Model trained and saved to models/emotion_lstm_model.h5")
