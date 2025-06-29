import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

try:
    # Use microphone as input
    with sr.Microphone() as source:
        print("🎤 Say something! I'm listening...")
        
        # Adjust for ambient noise with longer duration
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # Listen with timeout and phrase limit
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
    
    # Recognize speech using Google Web Speech API
    try:
        text = recognizer.recognize_google(audio)
        print("📝 You said:", text)
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
    except sr.RequestError as e:
        print(f"❌ Could not request results; {e}")

except Exception as e:
    print(f"❌ Microphone error: {e}")

# Transcribe from an audio file (e.g., WAV format)
file_path = "Acts_1-1.wav"  # Replace with your audio file path
 
try:
    with sr.AudioFile(file_path) as source:
        # 2. Adjust for potential noise in the recording
        recognizer.adjust_for_ambient_noise(source)
        
        # 3. Read the entire audio file
        audio_data = recognizer.record(source)
        
        try:
            # 4. Try transcription
            text = recognizer.recognize_google(audio_data)
            print("📝 Transcription:", text)
        except sr.UnknownValueError:
            print("❌ Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"❌ Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

except Exception as e:
    print(f"❌ Error processing audio file: {e}")