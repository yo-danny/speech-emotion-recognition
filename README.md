# Speech Emotion Recognition (SER)

## 📌 Project Overview
Speech Emotion Recognition (SER) is the act of identifying human emotions and affective states directly from speech. This project leverages the fact that vocal characteristics—such as tone, pitch, and energy—are deeply influenced by our emotional state.

This repository implements and compares three different machine learning and deep learning architectures to classify emotions from .wav audio files using acoustic features.

## 🎯 Objectives
Acoustic Feature Extraction: Extract robust time-frequency features from raw audio using the librosa library.

Comparative Analysis: Compare the performance of traditional Machine Learning (XGBoost), Artificial Neural Networks (MLP), and Deep Learning (CNN) architectures.

Emotion Classification: Train models to accurately recognize specific emotional states (e.g., Happy, Sad, Calm, Angry) from the RAVDESS dataset.

Performance Evaluation: Assess and document model performance using accuracy scores, confusion matrices, and classification reports.

## ⚙️ How It Works

### 1. Feature Extraction

Raw audio waveforms cannot be fed directly into machine learning models. We utilize librosa to transform raw signal data into a compact, 1D statistical feature vector by calculating the time-series mean of the following six acoustic properties:

- MFCC (Mel Frequency Cepstral Coefficients): Maps sound to human auditory perception, capturing the phonetic and timbral content of the voice (40 coefficients).
- Chroma STFT: Analyzes harmonic and tonal content by projecting the energy spectrum onto the 12 traditional pitch classes.
- Mel Spectrogram: Provides a mathematically scaled frequency-over-time representation of the audio signal, optimized for human hearing sensitivity.
- Spectral Centroid: Indicates the "center of mass" of the spectrum, effectively representing the "brightness" or average frequency of the sound.
- Spectral Rolloff: Marks the frequency threshold below which a specific percentage (usually 85%–95%) of the total spectral energy lies, indicating the decay of the signal.
- Zero-Crossing Rate (ZCR): Measures how often the audio signal crosses the zero-amplitude axis. This is highly effective for distinguishing between voiced/unvoiced speech and detecting high-energy, noisy emotional states (like anger or panic).

### 2. Repository Structure & Models

The project is split into two distinct Jupyter Notebooks to isolate and evaluate different modeling paradigms:

#### 📊 Notebook 1: Baseline Architecture (MLP)
Model: MLPClassifier (Multi-layer Perceptron) via scikit-learn.

Approach: A feedforward artificial neural network trained on flattened, aggregated statistical feature vectors (mean of MFCCs, Chroma, and Mel features). It serves as our baseline pattern recognition model.

#### 🚀 Notebook 2: Advanced Architectures (CNN & XGBoost)
Model A: 1D/2D CNN (Convolutional Neural Network) * Built using TensorFlow/Keras.

Approach: Treats feature matrices (like the Mel Spectrogram) as spatial or sequential data, using convolutional layers to automatically learn local, translation-invariant acoustic patterns.

Model B: XGBoost (Extreme Gradient Boosting)

Built using the xgboost library.

Approach: A powerful ensemble of gradient-boosted decision trees trained on the tabular feature vectors. Ideal for handling non-linear relationships with high computational efficiency.

## 📊 The Dataset
This project utilizes the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. It contains emotional speech recordings from professional actors speaking with various emotional expressions (Calm, Happy, Sad, Angry, Fearful, Surprise, and Disgust).

## 🛠️ Tech Stack

- Language: Python 3.x
- Audio Processing: librosa, soundfile
- Machine Learning & Tuning: scikit-learn, xgboost
- Deep Learning: tensorflow / keras
- Data Manipulation & Visualization: numpy, pandas, matplotlib, seaborn

## 🔄 The Pipeline

- Load Data: Use glob and soundfile to parse and read the RAVDESS directory structure.
- Extract Features: Transform raw audio data into structured numerical matrices.
- Train/Test Split: Partition the dataset into training and testing sets (e.g., 80/20 split) to ensure unbiased evaluation.
- Model Training: Fit the MLP, CNN, and XGBoost models on the training data.
- Evaluation & Comparison: Generate accuracy metrics and confusion matrices to determine which architecture performs best on speech emotion tasks.
