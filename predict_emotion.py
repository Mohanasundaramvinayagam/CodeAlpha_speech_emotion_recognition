import librosa
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

def predict_emotion(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    try:
        # Load model and encoder
        model = load_model("models/emotion_lstm_model.h5")
        with open("models/label_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
    except Exception as e:
        print(f"❌ Model or encoder loading failed: {e}")
        return

    try:
        # Load and preprocess audio
        signal, sr = librosa.load(file_path, sr=22050, mono=True)

        # Pad if too short
        if len(signal) < sr:
            pad_width = sr - len(signal)
            signal = np.pad(signal, (0, pad_width), mode='constant')

        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
        mfccs_mean = mfccs.mean(axis=1)
        x_input = np.array(mfccs_mean).reshape(1, 1, 40)

    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return

    try:
        predictions = model.predict(x_input)
        predicted_index = np.argmax(predictions)
        predicted_label = encoder.inverse_transform([predicted_index])[0]

        print("----- Emotion Recognition Result -----")
        print(f"🎧 File Name        : {file_path}")
        print(f"🎭 Predicted Emotion: {predicted_label}")
        print(f"📈 Confidence       : {np.max(predictions) * 100:.2f}%")
        print("--------------------------------------")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")

# Example usage
if __name__ == "__main__":
    test_file = "test_audio.wav"  # Change to your audio file
    predict_emotion(test_file)
