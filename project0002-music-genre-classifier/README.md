### Description:

A music genre classifier uses features extracted from audio files (such as tempo, pitch, spectral features) to classify songs into genres like rock, classical, jazz, pop, etc. In this project, we extract features using LibROSA and train a Random Forest classifier to predict genres using the GTZAN dataset or similar audio datasets.

- Extracts meaningful audio features using LibROSA
- Trains a classifier for multi-class genre prediction
- Evaluates with accuracy and confusion matrix

## Music Genre Classification using Random Forest

This project demonstrates how to build a simple music genre classification system using audio feature extraction with `librosa` and a `RandomForestClassifier` from `scikit-learn`.

---

### 1. **Overview of the Code**

The program performs the following steps:

#### a. **Feature Extraction (`extract_features`)**

* Loads a 30-second clip from an audio file using `librosa.load()`.
* Extracts three types of audio features:

  * **MFCCs (Mel Frequency Cepstral Coefficients)**: 13 coefficients that represent the timbral texture of the audio.
  * **Chroma Features**: Represent the intensity of each of the 12 different pitch classes.
  * **Spectral Contrast**: Measures the difference in amplitude between peaks and valleys in a sound spectrum.
* Takes the **mean across time** for each feature, resulting in a fixed-length feature vector.
* Combines them using `np.hstack()` into a single feature array.

#### b. **Dataset Loading**

* Assumes a folder named `genres` containing subfolders per genre (e.g., `blues`, `jazz`, `rock`).
* Iterates over each `.mp3` file in each genre folder.
* Extracts features and stores them with the corresponding genre label.

#### c. **Training the Model**

* Converts the feature list and labels into numpy arrays `X` and `y`.
* Splits the dataset into training and testing sets using `train_test_split()` (80% train, 20% test).
* Trains a `RandomForestClassifier` with 100 trees.

#### d. **Prediction and Evaluation**

* Predicts genres on the test set.
* Generates a classification report and prints precision, recall, f1-score, and support for each genre.
* Computes and plots a **confusion matrix** using Seaborn heatmap.

---

### 2. **Understanding the Results**

#### a. **Classification Report**

* Shows how well the model performed per genre.
* **Precision**: % of correct predictions for a genre out of all predicted as that genre.
* **Recall**: % of correct predictions out of actual occurrences of that genre.
* **F1-score**: Harmonic mean of precision and recall.
* **Support**: Number of actual occurrences in the test data.

#### b. **Confusion Matrix**

* Visual representation of prediction accuracy.
* Rows: True genres
* Columns: Predicted genres
* Diagonal elements: Correct predictions
* Off-diagonal: Misclassifications

---

### 3. **Interpretation and Use**

* A good model should show **high values on the diagonal** of the confusion matrix.
* Errors can often happen between similar-sounding genres (e.g., jazz vs blues).
* This project can be extended with:

  * Deep learning models (CNNs on spectrograms)
  * Data augmentation
  * Larger and more diverse datasets

---

### 4. **Dependencies**

Ensure you have the following Python packages:

```bash
pip install numpy librosa matplotlib scikit-learn seaborn
```

---

### 5. **Run Instructions**

```bash
python music_genre_classifier.py
```

Ensure your directory structure is like:

```
genres/
├── blues/
│   ├── file1.mp3
│   └── file2.mp3
├── jazz/
    └── ...
```

---

### 6. **Output Example**

```
Classification Report:
              precision    recall  f1-score   support

       blues       0.89      0.85      0.87        20
        jazz       0.86      0.90      0.88        20
        rock       0.80      0.75      0.77        20
        ...
```

---

This simple pipeline offers a strong baseline for music genre classification. It uses interpretable features and a robust ensemble learning model for solid performance on structured audio datasets.
