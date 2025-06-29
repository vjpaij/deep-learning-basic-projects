### Description:

Age detection from images aims to estimate a person’s age based on facial features. In this project, we use a pre-trained deep learning model to predict the approximate age group from a face image. This is often used in demographics analysis, targeted advertising, or access control systems.

- Detects faces using OpenCV's DNN face detector
- Estimates age using a pre-trained CNN model (Caffe)
- Works on single or multiple faces in an image

## Age Detection Using OpenCV and Pre-trained Caffe Models

### Overview

This script detects human faces in an image and predicts the age group of each detected face using pre-trained deep learning models. It utilizes OpenCV's DNN module to load and run models trained using the Caffe deep learning framework.

---

### Code Explanation

```python
import cv2
import numpy as np
```

* **cv2**: OpenCV library used for image processing and working with deep neural networks.
* **numpy**: Library used for numerical operations, particularly for working with arrays.

```python
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
```

* These are the age groups that the model can classify into.

#### Load Pre-trained Models

```python
age_model = cv2.dnn.readNetFromCaffe(...)
face_model = cv2.dnn.readNetFromCaffe(...)
```

* **age\_model**: A neural network trained to predict age from face images.
* **face\_model**: A face detection model based on SSD (Single Shot Detector) with ResNet-10.
* Both models are loaded from remote URLs using their `.prototxt` and `.caffemodel` files.

#### Load Input Image

```python
image = cv2.imread(cv2.samples.findFile("messi5.jpg"))
(h, w) = image.shape[:2]
```

* Reads the input image and extracts its dimensions.

#### Face Detection

```python
blob = cv2.dnn.blobFromImage(...)
face_model.setInput(blob)
detections = face_model.forward()
```

* Converts the image into a format required by the neural network (`blobFromImage`).
* Runs forward pass on the face detection model to get face bounding boxes.

#### Process Each Detection

```python
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.6:
        ...
```

* Loops through all detected faces.
* Only considers detections with confidence greater than 60%.

#### Extract and Preprocess Face

```python
face = image[startY:endY, startX:endX]
face_blob = cv2.dnn.blobFromImage(face, ...)
```

* Crops the face region from the original image.
* Preprocesses the face to feed into the age prediction model.

#### Age Prediction

```python
age_model.setInput(face_blob)
age_preds = age_model.forward()
age = AGE_BUCKETS[age_preds[0].argmax()]
confidence_score = age_preds[0].max()
```

* Performs forward pass on the age model to get prediction probabilities.
* Picks the age group with the highest probability.
* Extracts confidence score (highest probability).

#### Annotate the Image

```python
text = f"Age: {age} ({confidence_score*100:.1f}%)"
cv2.rectangle(...)
cv2.putText(...)
```

* Draws a bounding box around the detected face.
* Displays the predicted age group and confidence score.

#### Show the Output

```python
cv2.imshow("Age Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

* Displays the final image with annotations.

---

### Result/Prediction Interpretation

* **Age Bucket**: The model outputs the most probable age range the person belongs to.
* **Confidence Score**: The model's certainty about the prediction. For instance, `Age: (25-32) (85.4%)` means the model predicts the person is aged 25–32 with 85.4% confidence.

---

### Applications

* Real-time age prediction in surveillance systems.
* Demographic analysis in retail or marketing.
* Face analytics in security systems.

---

### Notes

* Make sure the image file path is correct (replace `"messi5.jpg"` with your actual image path).
* Internet is required to download the models if they are not already cached.
* Accuracy may vary based on image quality, lighting, and face visibility.

---

### Dependencies

* OpenCV with DNN module
* NumPy

Install via pip if needed:

```bash
pip install opencv-python numpy
```

---

### Sample Output

![Example Output](example_output.png)

Shows a face with a green box labeled like:

```
Age: (25-32) (85.4%)
```

This README serves as a guide to understanding and running the age detection script using OpenCV and Caffe models.


