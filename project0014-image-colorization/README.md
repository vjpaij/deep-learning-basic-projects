### Description:

Image colorization is the process of automatically adding color to grayscale images using AI. In this project, we build a deep learning model (a simplified CNN or U-Net) that learns to predict the color components of an image from its grayscale version. For quick prototyping, we’ll demonstrate using OpenCV and a pre-trained Caffe model for colorization.

- Takes grayscale input and predicts color components
- Uses pre-trained deep learning model (Caffe-based) via OpenCV
- Produces visually appealing colorized images

## Image Colorization using OpenCV and Caffe

This project demonstrates how to use a pre-trained deep learning model to automatically colorize grayscale images using OpenCV's DNN module and a Caffe model.

---

### 👨‍💻 Code Walkthrough

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

* **cv2**: OpenCV for computer vision tasks.
* **numpy**: For matrix operations.
* **matplotlib**: For visualizing the input and output images.

```python
prototxt = cv2.samples.findFile("dnn/colorization_deploy_v2.prototxt")
model = cv2.samples.findFile("dnn/colorization_release_v2.caffemodel")
pts = cv2.samples.findFile("dnn/pts_in_hull.npy")
```

* Load the **model configuration (`prototxt`)**, **pre-trained weights (`caffemodel`)**, and **cluster centers (`pts_in_hull.npy`)**.

```python
net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts_data = np.load(pts)
```

* Load the model into memory and read the cluster center data for color prediction.

```python
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts_data.astype(np.float32).reshape(2, 313, 1, 1)]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
```

* Initialize layers `class8_ab` and `conv8_313_rh` with color cluster centers and a fixed scaling factor.

```python
img_path = cv2.samples.findFile("messi5.jpg")
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
```

* Load a **grayscale image** and convert it to **LAB color space**, which separates lightness (L) from color (A, B).

```python
L = img_lab[:, :, 0]
L_resized = cv2.resize(L, (224, 224))
L_resized = L_resized - 50
```

* Extract the **L channel** (lightness), resize it to match the model's input (224x224), and **mean-center** it by subtracting 50.

```python
net.setInput(cv2.dnn.blobFromImage(L_resized))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
```

* Pass the L channel through the model to get the **predicted A and B channels**, which represent color.

```python
ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
```

* Resize the predicted AB channels to match the original image size and concatenate them with the original L channel.

```python
colorized_bgr = cv2.cvtColor(colorized.astype("float32"), cv2.COLOR_LAB2BGR)
colorized_bgr = np.clip(colorized_bgr, 0, 1)
```

* Convert the LAB image back to **BGR format**, clip values between \[0, 1] to ensure display compatibility.

```python
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Grayscale")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Colorized")
plt.imshow(colorized_bgr)
plt.axis("off")
plt.tight_layout()
plt.show()
```

* Plot and compare the **input grayscale image** and the **colorized output**.

---

### 📊 Result/Prediction Explained

* The output is a **colorized version** of the grayscale input image.
* The model was trained on a large dataset and has learned how natural colors usually appear.
* **Prediction**: The model doesn't restore original colors but **guesses plausible colors** based on patterns.
* **Result quality**: Works well for natural scenes (humans, objects, nature), but may struggle with abstract or synthetic content.

---

### 🔹 How to Use This

1. Install dependencies:

   ```bash
   pip install opencv-python numpy matplotlib
   ```
2. Download the following files into dnn folder:
   "https://www.kaggle.com/code/devarshpatel23/demo-of-black-to-color"
   * `colorization_deploy_v2.prototxt`
   * `colorization_release_v2.caffemodel`
   * `pts_in_hull.npy`
   * Your target grayscale image (e.g., `.jpg`)
3. Run the script and observe the colorization output.

---

### 📈 Summary

This script shows how deep learning can bring black-and-white images to life using a pre-trained colorization model. It's a great example of applying transfer learning to vision tasks using OpenCV and Caffe.

