### Description:

A text-to-speech (TTS) system converts written text into spoken audio. In this project, we use Python’s pyttsx3 library to create a simple TTS engine that reads text aloud. This is useful for building voice assistants, accessibility tools, or audiobook generators.

- Converts text into speech using offline TTS engine
- Configures speech rate, volume, and voice
- Works offline, no internet required

## Text-to-Speech (TTS) with `pyttsx3`

This Python script uses the `pyttsx3` library to convert a given text into speech using a text-to-speech (TTS) engine. Below is a detailed explanation of the code, the logic behind each part, and what the result means.

---

### Code Explanation

```python
import pyttsx3
```

* **Purpose**: Imports the `pyttsx3` module, a text-to-speech conversion library in Python.
* **Why**: `pyttsx3` is a Python library that works offline and supports multiple TTS engines.

```python
engine = pyttsx3.init()
```

* **Purpose**: Initializes the TTS engine.
* **Why**: This creates an engine instance that interfaces with the underlying speech engine (e.g., SAPI5 on Windows or NSSpeechSynthesizer on macOS).

```python
engine.setProperty('rate', 150)
```

* **Purpose**: Sets the speed (rate) of speech to 150 words per minute.
* **Why**: Adjusting the rate allows control over how fast the TTS engine reads the text. 150 is a moderate and understandable speed.

```python
engine.setProperty('volume', 1.0)
```

* **Purpose**: Sets the volume to the maximum level (1.0).
* **Range**: Between 0.0 (silent) and 1.0 (full volume).

```python
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
```

* **Purpose**: Selects a voice from the available list.
* **How**: `voices[0]` selects the first available voice. You can change the index to try different voices (e.g., male/female or different accents).

```python
text = "Hello! I am your AI assistant. How can I help you today?"
```

* **Purpose**: This is the text that will be converted to speech.

```python
engine.say(text)
```

* **Purpose**: Queues the `text` to be spoken.

```python
engine.runAndWait()
```

* **Purpose**: Runs the speech engine and waits until the speech is finished.
* **Why**: This ensures the script doesn’t end before the text is spoken.

---

### Output / Result

* The script produces audible speech from your computer speakers.
* In this case, it will say: **"Hello! I am your AI assistant. How can I help you today?"**

### What It Means

* **Prediction**: Not applicable here, as this is not a predictive model.
* **Score/Report**: Not applicable; there’s no accuracy or evaluation metric in a TTS script.
* **Result**: The main outcome is the successful vocalization of the text input.

### Use Cases

* Personal voice assistants
* Reading documents aloud
* Interactive voice response (IVR) systems
* Accessibility support for visually impaired users

---

### Note

* You can list all available voices with a simple loop:

```python
for voice in voices:
    print(voice.id)
```

* You can change the voice index to switch between male/female or different language voices depending on your OS and installed voices.

---

### Installation

To run this script, install `pyttsx3` if not already installed:

```bash
pip install pyttsx3
```
