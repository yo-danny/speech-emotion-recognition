# Speech Emotion Recognition (SER)

## 📌 Project Overview

Speech Emotion Recognition (SER) is the process of identifying human emotions and affective states directly from speech. It leverages the fact that vocal characteristics—such as tone, pitch, and energy—are deeply influenced by our emotional state.

This project implements a machine learning pipeline to classify emotions from .wav audio files using the MLPClassifier (Multi-layer Perceptron).

## 🎯 Objectives

- Extract acoustic features from audio using the librosa library.
- Analyze and process the RAVDESS dataset.
- Train a Deep Learning model to recognize specific emotional states (e.g., Happy, Sad, Calm, Angry).
- Evaluate model performance using accuracy metrics.

## 🛠️ Tech Stack

- Language: Python 3.x
- Audio Processing: librosa, soundfile
- Machine Learning: scikit-learn (MLPClassifier)
- Data Manipulation: numpy, pandas
- Environment: Jupyter Notebook / VS Code

## 📊 The Dataset

This project utilizes the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

## ⚙️ How It Works

### 1. Feature Extraction

Audio cannot be fed directly into an MLP. We extract three primary features:

- MFCC (Mel Frequency Cepstral Coefficients): Represents the short-term power spectrum of sound.
- Chroma: Relates to the 12 different pitch classes.
- Mel Spectrogram: Provides a visual representation of the spectrum of frequencies.

### 2. Model Architecture

We use a Multi-layer Perceptron (MLP) Classifier. This is a feedforward artificial neural network that is particularly effective for pattern recognition in structured feature sets.

### 3. Pipeline

- Load Data: Use glob and soundfile to read the RAVDESS files.
- Extract Features: Transform raw audio into a numerical feature vector.
- Train/Test Split: Standard 80/20 or 75/25 split using train_test_split.
- Training: Fit the MLPClassifier to the training data.
- Evaluation: Predict emotions on the test set and calculate the accuracy score.
