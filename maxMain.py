import os
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# Path to the dataset
dataset_path = "./deam_dataset"

# Load annotations
annotations_file = "./deam_dataset/DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
annotations = pd.read_csv(annotations_file)

# Fix column names that have spaces at the beginning
annotations.columns = annotations.columns.str.strip()
print("\nColumns after correction:", annotations.columns.tolist())

# Audio directory
audio_dir = "./deam_dataset/DEAM_audio/MEMD_audio"

# List all available audio files
audio_files = {}
for root, dirs, files in os.walk(audio_dir):
    for file in files:
        if file.endswith(".mp3") or file.endswith(".wav"):
            # Extract ID from filename (without extension)
            file_id = os.path.splitext(file)[0]
            audio_files[file_id] = os.path.join(root, file)

print(f"Number of audio files found: {len(audio_files)}")
print(f"Examples of available IDs: {list(audio_files.keys())[:5]}")

# Enhanced Feature extraction function
def extract_features(audio_path, n_mfcc=20, n_fft=2048, hop_length=512, duration=30):
    """Extract audio features from a file with enhanced feature set"""
    try:
        # Load audio file (limit to first duration seconds if specified)
        if duration:
            y, sr = librosa.load(audio_path, sr=None, duration=duration)
        else:
            y, sr = librosa.load(audio_path, sr=None)
        
        # === Temporal features ===
        # RMS energy
        rms = np.mean(librosa.feature.rms(y=y))
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        
        # === Spectral features ===
        # MFCCs (timbral features)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)  # Add standard deviation for temporal variation
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        
        # === Harmonic features ===
        # Chroma features (harmonic content)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        
        # === Rhythm features ===
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Onset strength 
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_strength_mean = np.mean(onset_env)
        
        # === Combine all features ===
        features = np.hstack([
            mfcc_mean, mfcc_std, 
            chroma,
            spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_contrast,
            rms, zcr, tempo, onset_strength_mean
        ])
        
        return features
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Prepare data
X = []  # Features
y = []  # Labels (valence and arousal)
song_ids_processed = []  # To track which songs have been processed

# Process maximum number of files
MAX_SAMPLES = len(audio_files)  # Adjust based on your computing power

# Track processing time
start_time = time.time()

# Process audio files matching annotations
for index, row in annotations.iterrows():
    # Convert ID to integer then string
    song_id = str(int(row['song_id']))
    
    if song_id in audio_files:
        audio_path = audio_files[song_id]
        features = extract_features(audio_path)
        if features is not None:
            X.append(features)
            y.append([row['valence_mean'], row['arousal_mean']])
            song_ids_processed.append(song_id)
    else:
        if index < 20:  # Only print first few missing files to avoid clutter
            print(f"Audio file not found for song_id {song_id}")
    
    # Show progress
    if index % 20 == 0:
        print(f"Processing: {index}/{len(annotations)}")
    
    # Limit number of samples
    if len(X) >= MAX_SAMPLES:
        break

# Report processing time
processing_time = time.time() - start_time
print(f"Feature extraction completed in {processing_time:.2f} seconds")

# Check if we have data
X = np.array(X) if X else np.array([])
y = np.array(y) if y else np.array([])

print(f"\nNumber of samples extracted: {len(X)}")
print(f"Feature vector dimension: {X.shape[1] if len(X) > 0 else 0}")
print(f"Examples of processed songs: {song_ids_processed[:5]}")

if len(X) > 0:
    # Split data with stratification based on binned emotion values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Print dataset sizes
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the processed data and scaler for later use
    import joblib
    
    # Create a directory for the processed data if it doesn't exist
    os.makedirs('processed_data', exist_ok=True)
    
    # Save the processed data and scaler
    np.save('processed_data/X_train.npy', X_train)
    np.save('processed_data/X_test.npy', X_test)
    np.save('processed_data/y_train.npy', y_train)
    np.save('processed_data/y_test.npy', y_test)
    np.save('processed_data/X_train_scaled.npy', X_train_scaled)
    np.save('processed_data/X_test_scaled.npy', X_test_scaled)
    joblib.dump(scaler, 'processed_data/feature_scaler.pkl')
    
    print("\nData processing completed and saved to 'processed_data' directory.")
    print("You can now use this data to train your models.")
    
    # Visualize the distribution of emotional values in the dataset
    plt.figure(figsize=(12, 6))
    
    # Plot valence distribution
    plt.subplot(1, 2, 1)
    plt.hist(y[:, 0], bins=20, alpha=0.7)
    plt.title('Valence Distribution')
    plt.xlabel('Valence')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Plot arousal distribution
    plt.subplot(1, 2, 2)
    plt.hist(y[:, 1], bins=20, alpha=0.7)
    plt.title('Arousal Distribution')
    plt.xlabel('Arousal')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('emotion_distribution.png')
    plt.show()
    
    # Visualize emotion space (valence-arousal plane)
    plt.figure(figsize=(10, 8))
    plt.scatter(y[:, 0], y[:, 1], alpha=0.6)
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.title('Emotion Space Distribution')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=np.mean(y[:, 1]), color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=np.mean(y[:, 0]), color='r', linestyle='--', alpha=0.5)
    plt.savefig('emotion_space.png')
    plt.show()
    
else:
    print("No data was extracted. Check the dataset structure.")