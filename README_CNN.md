# Speech Emotion Recognition (SER)

## 🛑 Project Overview

Speech Emotion Recognition (SER) is the process of identifying human emotions and affective states directly from speech. It leverages the fact that vocal characteristics—such as tone, pitch, and energy—are deeply influenced by our emotional state.

This project implements a machine learning pipeline to classify emotions from .wav audio files using a Deep Learning model built with Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) layers.

## 🎯 Objectives

- Extract acoustic features from audio using the librosa library, including MFCC, Chroma, Mel Spectrogram, Spectral Centroid, Spectral Rolloff, and Zero-Crossing Rate.
- Analyze and process the RAVDESS dataset.
- Train a Deep Learning model (CNN-LSTM) to recognize specific emotional states (e.g., Happy, Sad, Calm, Angry).
- Evaluate model performance using accuracy, precision, recall, and F1-score metrics.

## 🔨 Tech Stack

- Language: Python 3.x
- Audio Processing: librosa, soundfile
- Machine Learning: Keras (TensorFlow backend), scikit-learn
- Data Manipulation: numpy
- Environment: Jupyter Notebook / Google Colab

## 📊 The Dataset

This project utilizes the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

## ⚙️ How It Works

### 1. Feature Extraction

Audio cannot be fed directly into a neural network. We extract several primary features:

- **MFCC (Mel Frequency Cepstral Coefficients)**: Represents the short-term power spectrum of sound, capturing the timbre of the audio.
- **Chroma**: Relates to the 12 different pitch classes, indicating harmonic and tonal content.
- **Mel Spectrogram**: Provides a visual representation of the spectrum of frequencies over time, mimicking human hearing perception.
- **Spectral Centroid**: Indicates the ‘brightness’ of the sound, representing the average frequency.
- **Spectral Rolloff**: Measures the frequency below which a specified percentage of the total spectral energy resides, indicating spectral shape.
- **Zero-Crossing Rate**: Counts the number of times the waveform changes sign, useful for characterizing percussive sounds or noise.

### 2. Model Architecture

We use a Convolutional Neural Network (CNN) combined with Long Short-Term Memory (LSTM) layers. This architecture is particularly effective for sequential data like audio features, allowing the model to learn both local patterns (CNN) and long-range dependencies (LSTM). The model is designed with:
- A `Conv1D` layer for capturing local features across the audio features.
- A `MaxPooling1D` layer for downsampling.
- An `LSTM` layer to process sequential dependencies.
- `Dense` layers with `softmax` activation for final emotion classification.

### 3. Pipeline

- **Load Data**: Use `glob` and `soundfile` to read the RAVDESS `.wav` files.
- **Extract Features**: Transform raw audio into a numerical feature vector using the `extract_feature` function.
- **Train/Test Split**: Standard 75/25 split using `train_test_split`.
- **Encoding Labels**: Convert categorical emotion labels into numerical format using `LabelBinarizer`.
- **Training**: Compile and fit the Keras CNN-LSTM model to the training data, using `categorical_crossentropy` loss and `adam` optimizer.
- **Evaluation**: Predict emotions on the test set and calculate accuracy, precision, recall, and F1-score. A confusion matrix is also generated to visualize performance.

The current model achieves an accuracy of {:.2f}% on the test set, indicating room for further optimization and hyperparameter tuning.

