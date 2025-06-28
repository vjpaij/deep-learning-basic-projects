### Description:

A face recognition system identifies or verifies a person by analyzing facial features. In this project, we use the face_recognition library (built on dlib) to implement face detection, face encoding, and face comparison, allowing us to recognize faces from images or a webcam feed.

- Uses pre-trained face encodings for recognition
- Detects and labels faces in real-time via webcam
- Leverages the simplicity and power of face_recognition library

## Face Recognition System using Python

This script implements a real-time face recognition system using the `face_recognition` and `OpenCV` libraries in Python. It compares faces from a live webcam feed against a set of known faces stored in a folder. Below is a detailed explanation of the code, its logic, and what the result/output means.

---

### 1. **Setup and Initialization**

```python
import face_recognition
import cv2
import numpy as np
import os
```

* **face\_recognition**: Simplifies facial recognition tasks like face detection, encoding, and matching.
* **cv2 (OpenCV)**: Handles image/video processing and display.
* **numpy**: For numerical operations (e.g., finding minimum distances).
* **os**: Interact with the file system (read image files).

---

### 2. **Load Known Faces**

```python
known_faces_dir = 'known_faces'
known_encodings = []
known_names = []

for file in os.listdir(known_faces_dir):
    img_path = os.path.join(known_faces_dir, file)
    try:
        img = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(img)
        if len(encoding) > 0:
            known_encodings.append(encoding[0])
            known_names.append(os.path.splitext(file)[0])
        else:
            print(f"No faces found in {file}")
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
```

* Loops through each image in the `known_faces` directory.
* Extracts **face encodings** (128-dimension vectors) representing facial features.
* Stores both encodings and corresponding names (derived from filenames).
* Handles errors gracefully (e.g., if no face is found).

```python
if not known_encodings:
    print("No valid face encodings found in known_faces directory.")
    exit()
```

* Stops the program if no known faces are loaded.

---

### 3. **Start Webcam for Real-Time Face Detection**

```python
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")
```

* Initializes webcam (`0` = default camera).
* Loop continues until user presses 'q'.

---

### 4. **Frame Processing Loop**

```python
ret, frame = cap.read()
```

* Captures one frame from webcam.

```python
rgb_frame = frame[:, :, ::-1]
small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
```

* Converts from BGR (OpenCV default) to RGB.
* Downscales the frame (25%) for faster processing.

```python
face_locations = face_recognition.face_locations(small_frame)
face_encodings = face_recognition.face_encodings(small_frame, face_locations)
```

* Detects all face positions and generates encodings.

---

### 5. **Compare with Known Faces**

```python
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
    name = "Unknown"

    if len(matches) > 0:
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]
```

* **compare\_faces** returns True/False for each known face based on tolerance.
* **face\_distance** gives a similarity score (lower = more similar).
* **argmin** finds the index of the closest match.
* If there's a match within tolerance, retrieves corresponding name; else displays "Unknown".

```python
    top, right, bottom, left = top*4, right*4, bottom*4, left*4
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
```

* Rescales coordinates (because frame was resized earlier).
* Draws a rectangle and name label on the frame.

---

### 6. **Display and Quit**

```python
cv2.imshow("Face Recognition", frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cap.release()
cv2.destroyAllWindows()
```

* Displays the annotated frame.
* Exits when 'q' is pressed.

---

### 7. **Output Meaning / Report / Score**

* The system shows **live video feed** with rectangles drawn around detected faces.
* Each face is labeled with a **name** if matched to a known face; otherwise labeled **"Unknown"**.
* The match is based on:

  * Boolean match using `compare_faces`
  * Numerical distance using `face_distance` (lower value = better match)
* The closest match (smallest distance) is selected using `argmin()`.
* No formal accuracy/score is printed, but recognition reliability depends on:

  * Quality and number of known face images
  * Lighting and clarity of webcam feed
  * Face angle/rotation

---

### Summary

This code is a functional and fast face recognition system suitable for real-time applications like surveillance, attendance systems, or personal security. With clear modularity, it can be extended to include features like access logs, face re-training, and confidence threshold tuning.

---

> **Note**: To improve accuracy, ensure:
>
> * Good quality images in `known_faces`
> * One face per image
> * Consistent lighting and face angles
