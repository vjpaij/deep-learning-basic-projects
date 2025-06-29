import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
 
# Sample multilingual dataset (text and language labels)
data = {
    'text': [
        'Bonjour, comment ça va?',       # French
        'Hello, how are you?',           # English
        'Hola, ¿cómo estás?',            # Spanish
        'Hallo, wie geht es dir?',       # German
        'Ciao, come stai?',              # Italian
        'Olá, tudo bem?',                # Portuguese
        'こんにちは、お元気ですか？',    # Japanese
        '안녕하세요, 잘 지내세요?',        # Korean
        'Привет, как дела?',             # Russian
        'नमस्ते, आप कैसे हैं?'           # Hindi
    ],
    'language': [
        'French', 'English', 'Spanish', 'German', 'Italian',
        'Portuguese', 'Japanese', 'Korean', 'Russian', 'Hindi'
    ]
}
 
df = pd.DataFrame(data)
 
# Vectorize text with character-level TF-IDF
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X = vectorizer.fit_transform(df['text'])
y = df['language']
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
 
# Predict and evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues')
plt.title("Language Identification - Confusion Matrix")
plt.xlabel("Predicted Language")
plt.ylabel("True Language")
plt.tight_layout()
plt.show()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
 
# Test on new input
test_sentence = "¿Dónde está la biblioteca?"
X_new = vectorizer.transform([test_sentence])
predicted_lang = model.predict(X_new)[0]
print(f"\n📄 Input: \"{test_sentence}\"\n🔤 Predicted Language: {predicted_lang}")