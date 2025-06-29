### Description:

A speech-to-text converter transforms spoken audio into written text. In this project, we use Python’s speech_recognition library to capture voice input from a microphone or audio file and transcribe it into text using Google Speech Recognition API.

- Captures speech using microphone or file input
- Converts speech into text using Google Speech API
- Easy and fast way to build voice-enabled apps

## 🎙️ Speech Recognition with Python

This script demonstrates how to perform speech recognition using the `speech_recognition` Python library. It can:

1. Capture live audio through the microphone.
2. Recognize speech from a local audio file.

---

### 📦 Libraries Used

```python
import speech_recognition as sr
```

The `speech_recognition` library provides a simple API to convert speech into text using various engines/APIs, including Google Web Speech API (used here).

---

## 🔁 Microphone-based Speech Recognition

```python
recognizer = sr.Recognizer()
```

* Initializes a recognizer object that manages speech recognition settings and logic.

```python
with sr.Microphone() as source:
```

* Opens the system microphone as the input source.

```python
recognizer.adjust_for_ambient_noise(source, duration=1)
```

* Calibrates the recognizer to ignore background noise. `duration=1` listens for 1 second to measure the ambient noise level.

```python
audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
```

* Records audio:

  * `timeout=5`: waits max 5 seconds for the user to start speaking.
  * `phrase_time_limit=5`: records up to 5 seconds of speech once the user starts speaking.

### 🧠 Speech Recognition via Google API

```python
text = recognizer.recognize_google(audio)
```

* Sends the captured audio to Google Web Speech API and converts it into text.

#### ✅ Output Cases

* If successful: prints the recognized text.
* If speech is unintelligible: `UnknownValueError`
* If Google API is unreachable: `RequestError`

#### 📌 Example Output

```
🎤 Say something! I'm listening...
📝 You said: Hello, this is a test
```

---

## 📂 Audio File-based Transcription

```python
file_path = "Acts_1-1.wav"
```

* Path to an existing `.wav` audio file.

```python
with sr.AudioFile(file_path) as source:
```

* Opens the audio file for processing.

```python
recognizer.adjust_for_ambient_noise(source)
```

* Calibrates for any consistent background noise in the file.

```python
audio_data = recognizer.record(source)
```

* Reads the entire content of the audio file.

```python
text = recognizer.recognize_google(audio_data)
```

* Sends the full audio content to Google's API for transcription.

### 📝 Output Example

```
📝 Transcription: The former account I made, O Theophilus, of all that Jesus began both to do and teach...
```

---

## ✅ Results/Report/Meaning

* **Recognized Text:** The main result is the spoken or recorded words converted into readable text.
* **Use Case:** Great for real-time voice interfaces or transcribing recordings.
* **Limitations:**

  * Relies on stable internet (Google API is cloud-based).
  * Quality of recognition depends on audio clarity and ambient noise.

---

## 📌 Tips

* Always adjust for ambient noise to improve recognition accuracy.
* Use `.wav` files with good sampling rates for better results.
* Handle exceptions to make the application user-friendly.

---

## 🧪 Sample Applications

* Voice Assistants
* Meeting Transcribers
* Accessibility tools for the hearing impaired
* Voice-controlled applications
