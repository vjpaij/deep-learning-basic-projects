### Description:

Activity recognition involves identifying physical activities (like walking, running, sitting) based on data from sensors such as accelerometers and gyroscopes. In this project, we use the popular UCI HAR dataset to train a machine learning classifier (e.g., Random Forest or LSTM) to recognize activities from time-series sensor data.

- Recognizes human physical activity using time-series sensor data
- Uses a Random Forest classifier on engineered features from accelerometers and gyroscopes
- Evaluates performance with confusion matrix and classification report

# Human Activity Recognition using Random Forest

This project performs human activity recognition using the UCI HAR (Human Activity Recognition) dataset. The model is trained using the Random Forest classification algorithm to classify different types of physical activities based on sensor data from smartphones.

## Code Explanation

### Libraries Used

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
```

* `pandas` and `numpy`: For handling and manipulating data.
* `sklearn.ensemble.RandomForestClassifier`: Implements the Random Forest algorithm.
* `sklearn.metrics`: To evaluate the model using classification metrics.
* `seaborn` and `matplotlib`: For data visualization.

### Load Data

```python
def load_har_data():
    X_train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset/train/X_train.txt", delim_whitespace=True, header=None)
    y_train = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset/train/y_train.txt", header=None)
    X_test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset/test/X_test.txt", delim_whitespace=True, header=None)
    y_test = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset/test/y_test.txt", header=None)
    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()
```

* Loads training and testing data from the UCI repository.
* `X_train`, `X_test`: Feature vectors derived from smartphone sensors (accelerometer and gyroscope).
* `y_train`, `y_test`: Corresponding activity labels.
* `.values.ravel()`: Converts labels to 1D numpy arrays.

### Activity Labels Mapping

```python
activity_labels = {
    1: "Walking",
    2: "Walking_Upstairs",
    3: "Walking_Downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying"
}
```

* Maps numerical class labels to human-readable activity names.

### Train the Model

```python
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
```

* Initializes a Random Forest with 100 decision trees.
* `random_state=42`: Ensures reproducibility.
* Trains the model using the training data.

### Predictions

```python
y_pred = clf.predict(X_test)
```

* Predicts the activity labels on the unseen test data.

### Classification Report

```python
print(classification_report(y_test, y_pred, target_names=[activity_labels[i] for i in sorted(activity_labels)]))
```

* Provides precision, recall, and F1-score for each activity class:

  * **Precision**: Of all predictions for a class, how many were correct.
  * **Recall**: Of all actual instances of a class, how many were correctly predicted.
  * **F1-score**: Harmonic mean of precision and recall.
  * **Support**: Number of actual occurrences of the class in the dataset.

### Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=activity_labels.values(), yticklabels=activity_labels.values())
plt.title("Confusion Matrix - Activity Recognition")
plt.xlabel("Predicted Activity")
plt.ylabel("True Activity")
plt.tight_layout()
plt.show()
```

* Displays a heatmap of the confusion matrix:

  * Diagonal values: Correct predictions.
  * Off-diagonal values: Misclassifications.
  * Useful for identifying which activities are often confused.

## Results Interpretation

* A high precision, recall, and F1-score across all activities indicates the model is reliable.
* The confusion matrix visually shows which activities the model struggles to distinguish.
* Example insight: If "Walking\_Upstairs" is frequently misclassified as "Walking", it may indicate overlapping sensor patterns.

## Conclusion

The Random Forest classifier performs well in recognizing human activities using sensor data from smartphones. This project serves as a good baseline for further research in activity recognition using wearable or mobile sensors.

## Possible Improvements

* Feature selection to reduce dimensionality.
* Hyperparameter tuning.
* Try deep learning models like CNN or LSTM for time-series analysis.
