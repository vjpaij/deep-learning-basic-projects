### Description:

A speaker recognition system identifies a person based on their unique voice characteristics. In this project, we use MFCC (Mel-frequency cepstral coefficients) extracted from audio files and build a machine learning model (like SVM or Random Forest) to classify speakers.

- Extracts voice features (MFCCs) to distinguish speakers
- Trains a machine learning model for classification
- Visualizes results using a confusion matrix

## Speaker Recognition using MFCC and SVM

### Overview

This project implements a basic **Speaker Recognition System** using **Mel Frequency Cepstral Coefficients (MFCC)** for audio feature extraction and **Support Vector Machine (SVM)** for classification. The objective is to correctly identify the speaker from a given `.wav` audio file.

---

### Code Explanation

#### Imports

```python
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
```

* `librosa`: For audio processing.
* `numpy`: For numerical operations.
* `matplotlib`, `seaborn`: For visualization.
* `sklearn`: For machine learning operations (model training, evaluation).

---

### Feature Extraction using MFCC

```python
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, duration=10)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)
```

* Loads the audio file (only first 10 seconds).
* Extracts 13 MFCC features.
* Returns the mean value of each MFCC across all time frames.
* MFCCs represent the short-term power spectrum of sound and are widely used in speech recognition.

---

### Dataset Preparation

```python
features = []
labels = []

for speaker in os.listdir("speakers"):
    for file in os.listdir(f"speakers/{speaker}"):
        if file.endswith(".wav"):
            try:
                mfcc = extract_mfcc(f"speakers/{speaker}/{file}")
                features.append(mfcc)
                labels.append(speaker)
            except Exception as e:
                print(f"Error with file: {e}")
```

* Iterates through all speakers and their `.wav` files.
* Extracts MFCC features for each audio file.
* Builds `features` (X) and `labels` (y) lists.

---

### Model Training

```python
X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)
```

* Converts feature and label lists into NumPy arrays.
* Splits data into 80% training and 20% testing.
* Trains a **Linear SVM Classifier**.

---

### Model Evaluation

```python
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

* Predicts speaker labels for the test set.
* Displays **precision**, **recall**, **f1-score**, and **accuracy** for each speaker:

  * **Precision**: How many selected items are relevant.
  * **Recall**: How many relevant items are selected.
  * **F1-score**: Harmonic mean of precision and recall.

---

### Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred, labels=speakers)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=speakers, yticklabels=speakers, cmap='Blues')
```

* Displays a heatmap of actual vs. predicted speakers.
* Each cell shows the number of samples classified from one speaker to another.
* Diagonal cells indicate correct predictions.

---

### Results Interpretation

* A high diagonal dominance in the confusion matrix indicates good classification accuracy.
* The classification report helps evaluate individual speaker recognition performance.
* Overall, the system serves as a simple but effective speaker identifier using MFCC and SVM.

---

### Requirements

Ensure the following Python packages are installed:

```bash
pip install librosa numpy matplotlib scikit-learn seaborn
```

---

### Directory Structure

```
speakers/
    alice/
        audio1.wav
        audio2.wav
    bob/
        audio1.wav
        audio2.wav
```

---

### Improvements

* Use more robust features like delta MFCCs.
* Explore deeper models (CNNs or LSTMs).
* Use full audio duration or segment into frames.
* Data augmentation for more robust models.
