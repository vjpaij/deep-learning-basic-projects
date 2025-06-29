### Description:

Part-of-Speech (POS) tagging is the process of labeling each word in a sentence with its grammatical role, such as noun, verb, adjective, etc. This is foundational in NLP for understanding syntax, sentence structure, and downstream tasks like parsing or entity recognition. In this project, we use spaCy to perform POS tagging on a given sentence.

- Tags each word in a sentence with its part of speech
- Helps in understanding sentence structure and grammar
- Uses spaCy's built-in linguistic pipeline with excellent performance

## Part-of-Speech Tagging with spaCy

This code demonstrates how to perform **Part-of-Speech (POS) tagging** using the `spaCy` NLP library in Python. POS tagging is a fundamental step in natural language processing (NLP), where each word in a sentence is classified into its grammatical role—such as noun, verb, adjective, etc.

### Code Explanation

```python
import spacy

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# Sample sentence
text = "The quick brown fox jumps over the lazy dog near the riverbank."

# Process text
doc = nlp(text)

# Print each word with its POS tag and explanation
print("Word\t\tPOS\t\tExplanation")
print("-" * 40)
for token in doc:
    print(f"{token.text:<10}\t{token.pos_:<10}\t{spacy.explain(token.pos_)}")
```

### How the Code Works

1. **Import spaCy**: Loads the `spacy` library for NLP.
2. **Load the Model**: Loads the small English model `en_core_web_sm`, which contains pre-trained pipelines for tokenization, POS tagging, parsing, and named entity recognition.
3. **Input Text**: The example sentence is a common English pangram.
4. **Processing**: `nlp(text)` processes the input and returns a `Doc` object, which is a container for tokens.
5. **Token Analysis**: Iterates over each `token` (word or punctuation) in the document and prints:

   * `token.text`: The original word.
   * `token.pos_`: The predicted POS tag.
   * `spacy.explain(token.pos_)`: A human-readable explanation of the POS tag.

### Example Output

```
Word        POS        Explanation
----------------------------------------
The         DET        determiner
quick       ADJ        adjective
brown       ADJ        adjective
fox         NOUN       noun
jumps       VERB       verb
over        ADP        adposition
the         DET        determiner
lazy        ADJ        adjective
dog         NOUN       noun
near        ADP        adposition
the         DET        determiner
riverbank   NOUN       noun
.           PUNCT      punctuation
```

### Result / Report / Score / Prediction Explained

* **Result**: Each word is assigned a POS tag.
* **Prediction**: These tags are inferred based on statistical and rule-based models trained on annotated text data.
* **Meaning**: POS tags help understand the grammatical structure and function of words, which is essential for tasks like parsing, named entity recognition, text classification, and question answering.

### Summary

This script illustrates how spaCy performs automatic POS tagging on a sentence, allowing further syntactic and semantic analysis. The output can be used to:

* Understand grammatical roles of words.
* Prepare data for more advanced NLP tasks.
* Extract linguistic features for downstream ML models.
