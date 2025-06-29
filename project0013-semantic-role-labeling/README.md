### Description:

Semantic Role Labeling (SRL) is the process of assigning roles to words or phrases in a sentence to determine who did what to whom, when, and how. It helps in understanding sentence meaning beyond grammar. In this project, we use the AllenNLP library to perform SRL on English text.

Example Output:
▶ Verb: gave
   Tags: B-ARG0 B-V B-ARG1 I-ARG1 B-ARG2 B-ARGM-TMP I-ARGM-TMP
This tells us:
- ARG0: Who? → John
- V: Verb → gave
- ARG1: What? → a book
- ARG2: To whom? → to Mary
- ARGM-TMP: When? → on her birthday

🧠 What This Project Demonstrates:
- Identifies semantic roles in sentences (like subject, object, time, etc.)
- Helps in understanding the meaning and intent behind statements
- Uses BERT-based pretrained model from AllenNLP

## Semantic Role Labeling with AllenNLP

This code demonstrates how to perform **Semantic Role Labeling (SRL)** using a pretrained BERT-based model from the AllenNLP library.

### 🔍 What is Semantic Role Labeling?

Semantic Role Labeling is a natural language processing task that identifies the predicate (usually a verb) in a sentence and determines the semantic roles of associated words or phrases (like "who" did "what" to "whom", "when", and "where").

---

### 📦 Code Explanation

```python
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
```

* **`Predictor`**: This is a high-level API in AllenNLP for running predictions using pre-trained models.
* **`allennlp_models.structured_prediction`**: This imports the SRL model architecture and configurations used by the predictor.

```python
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
```

* Loads a **pretrained BERT-based SRL model** from AllenNLP’s model zoo.
* The model is trained to identify verbs and tag words in a sentence with their respective semantic roles.

```python
sentence = "John gave a book to Mary on her birthday."
```

* The input sentence for which we want to analyze semantic roles.

```python
result = predictor.predict(sentence=sentence)
```

* This processes the sentence and returns a dictionary with semantic role labels for each verb.

```python
for verb in result['verbs']:
    print(f"\u25B6 Verb: {verb['verb']}")
    print(f"   Tags: {' '.join(verb['tags'])}")
```

* Iterates over each **verb** detected in the sentence.
* For each verb, prints the **semantic role tags** associated with each word.

---

### 📊 Sample Output Explanation

For the sentence:

```plaintext
John gave a book to Mary on her birthday.
```

You might see output like:

```plaintext
▶ Verb: gave
   Tags: B-ARG0 B-V B-ARG1 I-ARG1 B-ARG2 B-ARGM-TMP I-ARGM-TMP
```

#### What Do These Tags Mean?

* **B-ARG0**: Beginning of Argument 0 (the agent/doer) → *John*
* **B-V**: Beginning of Verb → *gave*
* **B-ARG1 / I-ARG1**: Beginning and inside of Argument 1 (the thing given) → *a book*
* **B-ARG2**: Beginning of Argument 2 (recipient) → *to Mary*
* **B-ARGM-TMP / I-ARGM-TMP**: Temporal modifier → *on her birthday*

These tags follow the **BIO** format (Beginning, Inside, Outside), labeling which parts of the sentence fulfill which semantic roles for the verb.

---

### ✅ Summary

* This script uses AllenNLP’s powerful SRL model to decompose a sentence into semantic roles.
* It identifies the verb(s) and labels arguments like agent, theme, recipient, and temporal modifiers.
* The output can be used to enrich NLP applications like question answering, information extraction, and machine reading comprehension.
