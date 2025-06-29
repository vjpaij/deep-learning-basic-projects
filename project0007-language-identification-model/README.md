### Description:

A language identification model determines the language of a given text input. It is commonly used in multilingual applications like translators, chatbots, and voice assistants. In this project, we build a simple text classification model using character-level TF-IDF features and a Logistic Regression classifier.

- Detects the language of a given sentence
- Uses character-level TF-IDF for robustness to short text
- Supports multiple languages easily

### Language Identification using Logistic Regression

This script demonstrates a simple multilingual language identification model using machine learning with character-level TF-IDF and logistic regression. It is designed to predict the language of a given sentence.

---

## ✨ What the Code Does (Step-by-Step Explanation)

### 1. **Dataset Creation**

```python
import pandas as pd
...
data = {
    'text': [...],
    'language': [...]
}
df = pd.DataFrame(data)
```

* A small sample dataset of 10 text phrases in different languages is defined.
* Each text is labeled with its corresponding language.

### 2. **Text Vectorization using TF-IDF**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X = vectorizer.fit_transform(df['text'])
```

* TF-IDF (Term Frequency-Inverse Document Frequency) is applied at the **character level**.
* Character n-grams (1 to 3 characters) are extracted, which helps in identifying language-specific patterns.
* The output `X` is a sparse matrix where each row represents a text and each column a character n-gram.

### 3. **Preparing Labels**

```python
y = df['language']
```

* Extracts the language labels from the DataFrame.

### 4. **Train-Test Split**

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

* The dataset is split into 70% training and 30% testing data.

### 5. **Training the Model**

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

* A logistic regression model is trained to classify the text into languages.
* `max_iter=1000` ensures convergence during training.

### 6. **Model Evaluation**

```python
from sklearn.metrics import classification_report, confusion_matrix
...
print(classification_report(y_test, y_pred))
```

* The model's predictions are compared to actual labels.
* Outputs:

  * **Precision**: How many selected items are relevant.
  * **Recall**: How many relevant items are selected.
  * **F1-Score**: Harmonic mean of precision and recall.

### 7. **Visualizing the Confusion Matrix**

```python
import seaborn as sns
import matplotlib.pyplot as plt
...
sns.heatmap(cm, annot=True, fmt='d', ...)
```

* The confusion matrix shows true vs predicted labels.
* Each row is the actual language; each column is the predicted language.

### 8. **Prediction on New Sentence**

```python
test_sentence = "¿Dónde está la biblioteca?"
X_new = vectorizer.transform([test_sentence])
predicted_lang = model.predict(X_new)[0]
```

* The model makes a prediction for a new input sentence.
* The predicted language is printed.

---

## 📊 Sample Output and Explanation

```
Classification Report:
              precision    recall  f1-score   support

      Hindi       1.00      1.00      1.00         1
    Italian       1.00      1.00      1.00         1
    Russian       1.00      1.00      1.00         1
    Spanish       1.00      1.00      1.00         1

Confusion Matrix: A heatmap showing perfect prediction (all diagonal entries are 1).

📄 Input: "¿Dónde está la biblioteca?"
🔤 Predicted Language: Spanish
```

**Explanation:**

* The classifier successfully predicted all test languages in this tiny dataset.
* For the test sentence in Spanish, the model correctly predicted `Spanish`.

---

## 💡 Key Takeaways

* Character-level TF-IDF is highly effective for language identification.
* Logistic regression is a simple yet effective model for text classification.
* Even with a small dataset, language-specific n-gram patterns can be learned.

---

## 🌐 Possible Extensions

* Add more samples per language for better generalization.
* Include more languages.
* Try other models: SVM, Naive Bayes, or Deep Learning (LSTM, BERT).
* Evaluate using metrics like accuracy, macro-F1 on larger datasets.

---

## 📁 Files Needed

No external files are needed; the dataset is created in-code.

---

## 🎓 Requirements

```bash
pip install pandas scikit-learn seaborn matplotlib
```
