# Music Emotion Prediction

Prediction of music emotions (Valence and Arousal) from audio features using the DEAM dataset.

## Dataset

This project uses the [DEAM (Database for Emotion Analysis in Music)](https://cvml.unige.ch/databases/DEAM/) dataset.

Download it and place it in the project root with the following structure:

```
deam_dataset/
├── DEAM_Annotations/
│   └── annotations/
│       └── annotations averaged per song/
│           └── song_level/
├── DEAM_audio/
│   └── MEMD_audio/
└── features/
    └── features/
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

- `main.py` — Main pipeline: trains Random Forest and neural network models on preextracted features from the dataset.
- `maxMain.py` — Alternative pipeline: extracts audio features directly from audio files using librosa, then trains models.

```bash
python main.py
```
