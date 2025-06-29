### Description:

Named Entity Recognition (NER) is the task of identifying and classifying entities in text into predefined categories such as person names, organizations, locations, dates, and more. In this project, we build a simple NER system using the spaCy library, which provides pre-trained models out of the box.

- Extracts real-world entities from unstructured text
- Uses pre-trained spaCy models for accurate recognition
- Identifies entities like PERSON, ORG, GPE, DATE, etc.

## Named Entity Recognition (NER) with spaCy

This code demonstrates how to perform **Named Entity Recognition (NER)** using the `spaCy` library in Python. NER is a fundamental task in Natural Language Processing (NLP) that involves identifying and classifying entities in text into predefined categories such as persons, organizations, locations, dates, etc.

---

### 🛠 Setup Instructions

1. Install spaCy (if not already installed):

```bash
pip install spacy
```

2. Download the English language model:

```bash
python -m spacy download en_core_web_sm
```

---

### 📄 Code Explanation

```python
import spacy
from spacy import displacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Barack Obama was born in Hawaii and served as the 44th President of the United States. He studied at Harvard University."

# Process text
doc = nlp(text)

# Print named entities
print("Named Entities:\n")
for ent in doc.ents:
    print(f"{ent.text:<30} \u27a4 {ent.label_}")

# Optional: Visualize entities in a browser
displacy.render(doc, style="ent", jupyter=False)
```

#### Step-by-Step Breakdown:

* **`import spacy`**: Imports the spaCy library for NLP tasks.
* **`from spacy import displacy`**: Imports spaCy's visualizer for rendering parsed content.
* **`spacy.load("en_core_web_sm")`**: Loads the small English model which contains pre-trained weights for tokenization, POS tagging, and NER.
* **`text = "..."`**: This is the sample input sentence.
* **`doc = nlp(text)`**: Passes the text through the NLP pipeline. The output is a processed `Doc` object.
* **`doc.ents`**: A list of named entities detected in the text.
* **Loop and print**: Displays each named entity and its label (like `PERSON`, `GPE`, `ORG`).
* **`displacy.render()`**: Opens a web browser and highlights named entities with colors and labels for easy visualization.

---

### 📊 Output Example:

```
Named Entities:

Barack Obama                  ➤ PERSON
Hawaii                        ➤ GPE
44th                          ➤ ORDINAL
President                     ➤ ORG
United States                ➤ GPE
Harvard University           ➤ ORG
```

---

### 🔍 Interpretation of Results

* **PERSON**: Recognized names of people. `Barack Obama` is classified correctly.
* **GPE (Geo-Political Entity)**: Represents countries, cities, or states. `Hawaii` and `United States` are classified as GPEs.
* **ORDINAL**: Indicates position in an ordered sequence. `44th` is the ordinal number here.
* **ORG (Organization)**: Refers to institutions, companies, etc. `President` and `Harvard University` are identified as organizations (though `President` might sometimes be misclassified depending on context).

---

### 🧠 Why This Matters

NER is critical in:

* Information extraction
* Question answering systems
* Resume parsing
* Document classification

Using spaCy simplifies this process by providing robust, pre-trained models and fast processing pipelines.

---

### 📈 Visualization (Optional)

If you run the script in a local Python environment (not in Jupyter), `displacy.render` will open a web browser with visual highlights for all detected entities.

> ✅ Use `style="ent"` to specifically visualize named entities.
