### Description:

Emotion recognition from text involves analyzing sentences or messages to classify them into emotional categories such as joy, sadness, anger, fear, etc. In this project, we build a text classification model using TF-IDF and Logistic Regression to detect emotions in short text samples.

- Classifies short text into emotional categories
- Uses TF-IDF features + Logistic Regression
- Evaluates with precision, recall, and confusion matrix

## Emotion Classification using Logistic Regression

This project demonstrates a simple emotion classifier using machine learning with Python. It processes text inputs (sentences expressing different emotions), vectorizes them using TF-IDF, and trains a logistic regression model to predict the emotion associated with each sentence.

### Code Explanation

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```

* **Imports**: Standard libraries for data manipulation (pandas), text feature extraction (TF-IDF), model training (logistic regression), model evaluation (classification report, confusion matrix), and visualization (matplotlib, seaborn).

```python
data = {
    'Text': [...],
    'Emotion': [...]
}
df = pd.DataFrame(data)
```

* **Dataset Creation**: A small sample dataset is defined, containing text sentences and their corresponding emotions (joy, anger, sadness, fear).

```python
df['Emotion_Label'] = df['Emotion'].astype('category').cat.codes
label_map = dict(enumerate(df['Emotion'].astype('category').cat.categories))
```

* **Label Encoding**: Emotions are converted to numerical labels (e.g., joy=2, anger=0, etc.) which are required for machine learning models.
* **Label Map**: Creates a reverse lookup dictionary to map label codes back to their original emotion names.

```python
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Text'])
y = df['Emotion_Label']
```

* **TF-IDF Vectorization**: Converts the text data into numerical features, removing common English stopwords. TF-IDF (Term Frequency-Inverse Document Frequency) emphasizes important words by penalizing commonly used ones.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

* **Train/Test Split**: 70% of data is used for training, and 30% for testing the model.

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

* **Model Training**: A logistic regression model is trained on the training data. `max_iter=1000` ensures convergence with small datasets.

```python
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_map.values()))
```

* **Prediction and Evaluation**:

  * `y_pred`: Predicted emotion labels for test data.
  * `classification_report`: Displays precision, recall, f1-score, and support for each emotion class.

```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='YlOrRd', fmt='d',
            xticklabels=label_map.values(), yticklabels=label_map.values())
plt.title("Emotion Classification - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
```

* **Confusion Matrix**: Visualizes how well the model is performing in terms of correct and incorrect predictions across emotion classes.

  * Rows = Actual emotions
  * Columns = Predicted emotions
  * Diagonal values = Correct predictions

### Understanding the Output

#### Classification Report

The output includes:

* **Precision**: How many selected items are relevant.
* **Recall**: How many relevant items are selected.
* **F1-score**: Harmonic mean of precision and recall.
* **Support**: Number of true instances for each label.

High scores indicate good model performance.

#### Confusion Matrix

* Provides a count of actual vs predicted classifications.
* Ideal result: high numbers along the diagonal, indicating correct predictions.

### Summary

This basic example shows how natural language can be converted into numerical features and classified using logistic regression. Despite its simplicity and small dataset, it sets the foundation for building more sophisticated emotion classifiers on larger corpora.
