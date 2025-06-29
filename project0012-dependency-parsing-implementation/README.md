### Description:

Dependency parsing analyzes the grammatical structure of a sentence by establishing relationships between "head" words and words that modify those heads. In this project, we use spaCy to perform dependency parsing and visualize how different words in a sentence relate to one another.

- Identifies grammatical dependencies (e.g., subject, object, modifiers)
- Reveals the syntactic structure of complex sentences
- Uses spaCy’s dependency parser for fast and accurate results

### spaCy Dependency Parsing Example

This Python script demonstrates how to use spaCy for dependency parsing of a natural language sentence. Dependency parsing is a process in natural language processing (NLP) where we analyze the grammatical structure of a sentence to establish relationships between words.

---

#### Code Breakdown and Explanation

```python
import spacy
from spacy import displacy

# Load English language model
nlp = spacy.load("en_core_web_sm")
```

* `spacy.load("en_core_web_sm")`: Loads a pre-trained small English language model. This model is trained on a dataset to understand word structure, grammar, and syntactic relations.

```python
# Input sentence
text = "The little girl saw a dog chasing a squirrel in the park."

# Process the sentence
doc = nlp(text)
```

* `text`: A sample input sentence to analyze.
* `doc = nlp(text)`: This runs the sentence through spaCy's pipeline, which includes tokenization, part-of-speech tagging, dependency parsing, and named entity recognition.

```python
# Print tokens with their dependency relationships
print(f"{'Token':<15} {'Dep':<15} {'Head':<15} {'POS':<10}")
print("-" * 60)
for token in doc:
    print(f"{token.text:<15} {token.dep_:<15} {token.head.text:<15} {token.pos_:<10}")
```

* This prints a table with columns:

  * `Token`: The actual word.
  * `Dep`: The type of syntactic relation it holds (dependency label).
  * `Head`: The word to which this token is syntactically connected (its parent in the tree).
  * `POS`: Part of Speech (e.g., NOUN, VERB, ADJ).

```python
# Visualize dependency tree in the browser
displacy.render(doc, style="dep", jupyter=False)
```

* `displacy.render`: A spaCy visualizer that displays the dependency tree structure in your default web browser.

---

#### Output Example

```
Token           Dep             Head            POS       
------------------------------------------------------------
The             det             girl            DET       
little          amod            girl            ADJ       
girl            nsubj           saw             NOUN      
saw             ROOT            saw             VERB      
a               det             dog             DET       
dog             dobj            saw             NOUN      
chasing         acl             dog             VERB      
a               det             squirrel        DET       
squirrel        dobj            chasing         NOUN      
in              prep            chasing         ADP       
the             det             park            DET       
park            pobj            in              NOUN      
.               punct           saw             PUNCT     
```

---

#### Explanation of Output

* **ROOT**: Main verb of the sentence is `saw`. Everything else builds around it.
* **nsubj (nominal subject)**: `girl` is the subject performing the action `saw`.
* **dobj (direct object)**: `dog` is the object being seen.
* **acl (clausal modifier of noun)**: `chasing` modifies `dog`, giving more information about what the dog is doing.
* **prep (preposition)** + **pobj (object of preposition)**: `in` introduces a prepositional phrase `in the park`, and `park` is the object of the preposition.

---

#### Report / Prediction / Score

This example does not involve classification, prediction, or scoring. Instead, it gives a **structural linguistic analysis** of the sentence, showing how words relate to one another.

* This kind of parsing is useful in applications like:

  * Question answering
  * Grammar checking
  * Machine translation
  * Text summarization

It allows downstream models to understand **who did what to whom, when, and where**, based on the syntactic relationships.

---

#### Visualization Output

The browser-based rendering (via `displacy`) shows a tree where:

* Arrows point from head to dependent tokens.
* Labels on arrows denote the grammatical relationship.
* The structure is visually intuitive for understanding sentence flow.

This is helpful in both debugging and interpreting complex NLP pipelines.

---

#### Summary

This script shows how spaCy can parse and visualize sentence structure, enabling deeper insights into grammatical composition. Though it doesn't involve ML "scores," it forms the backbone for many intelligent NLP systems by providing syntactic context.
