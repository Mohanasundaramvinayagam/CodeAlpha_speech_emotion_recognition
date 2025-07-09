import os
import pickle
import librosa
import pandas as pd
import numpy as np

def save_features():
    dataset_path = 'data/'  
    features = []
    errors = 0
    total_files = 0

    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                total_files += 1
                file_path = os.path.join(root, file)

                try:
                    signal, sr = librosa.load(file_path, sr=22050, mono=True)

                    
                    if len(signal) < sr:
                        pad_width = sr - len(signal)
                        signal = np.pad(signal, (0, pad_width), mode='constant')

                    
                    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
                    mfccs_mean = mfccs.mean(axis=1)

                    emotion_code = file.split("-")[2]  
                    label = emotion_map.get(emotion_code, "unknown")

                    if label != "unknown":
                        features.append({"features": mfccs_mean.tolist(), "label": label})
                except Exception as e:
                    print(f" Failed to process {file_path} — {e}")
                    errors += 1

   
    df = pd.DataFrame(features)
    os.makedirs("features", exist_ok=True)
    with open("features/mfcc_features.pkl", "wb") as f:
        pickle.dump(df, f)

    print(" Feature extraction complete")
    print(f" Files processed: {total_files}, Success: {len(df)}, Errors: {errors}")


if __name__ == "__main__":
    save_features()
