import face_recognition
import cv2
import numpy as np
import os

# Load known faces (assume folder "known_faces/name.jpg")
known_encodings = []
known_names = []

known_faces_dir = 'known_faces'  # Replace with your folder path

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

if not known_encodings:
    print("No valid face encodings found in known_faces directory.")
    exit()

# Load test image or use webcam
cap = cv2.VideoCapture(0)  # Use webcam; alternatively load image with cv2.imread()

print("Press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame from BGR (OpenCV) to RGB (face_recognition)
    rgb_frame = frame[:, :, ::-1]

    # Resize for faster processing
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

    # Detect faces and encode
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    # Compare with known encodings
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        name = "Unknown"

        if len(matches) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        # Scale back up face location
        top, right, bottom, left = top*4, right*4, bottom*4, left*4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display result
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()