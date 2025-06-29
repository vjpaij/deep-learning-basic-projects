### Description:

A Handwritten Equation Solver reads handwritten mathematical expressions from images, recognizes the characters using OCR, parses the equation, and solves it. This combines image processing, deep learning-based character recognition, and symbolic computation (e.g., using sympy).

- Uses OCR (Optical Character Recognition) to read handwritten math
- Converts math strings to symbolic expressions using sympy
- Solves algebraic equations automatically

# OCR Handwritten Equation Solver - README

This project demonstrates how to use computer vision and symbolic mathematics to solve handwritten equations from an image. It uses OpenCV for image preprocessing, Tesseract for OCR (Optical Character Recognition), and SymPy for mathematical parsing and solving.

---

## 🧠 Code Breakdown with Explanation

```python
import cv2
import pytesseract
from sympy import *
from sympy.parsing.sympy_parser import parse_expr

# Initialize symbolic math
x, y, z = symbols('x y z')
```

* **Libraries**:

  * `cv2` (OpenCV): Used for image reading and preprocessing.
  * `pytesseract`: OCR tool to extract text (handwritten or printed) from images.
  * `sympy`: Library for symbolic mathematics.
* **Symbol Initialization**: Declares the variables `x`, `y`, and `z` to be used in solving equations.

```python
image = cv2.imread("equation.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

* **Image Reading**: Loads the image `equation.png`.
* **Grayscale Conversion**: Converts the image to grayscale to simplify further processing.

```python
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, 11, 2)
```

* **Adaptive Thresholding**: Enhances image contrast, especially for handwritten text, improving OCR accuracy.

```python
custom_config = r'--oem 3 --psm 6'
extracted_text = pytesseract.image_to_string(gray, config=custom_config)
```

* **OCR Config**:

  * `--oem 3`: Use the default LSTM OCR engine.
  * `--psm 6`: Assume a single block of text.
* **OCR Execution**: Extracts the equation as a string from the image.

```python
print("📝 Extracted Equation:", extracted_text)
```

* Outputs the raw extracted text.

```python
try:
    equation = extracted_text.strip().replace(" ", "").replace("=", "==")

    lhs, rhs = equation.split("==")
    expr = Eq(parse_expr(lhs), parse_expr(rhs))
    solution = solve(expr)

    print("✅ Solution:", solution)
```

* **Sanitization**: Cleans up the extracted string by removing spaces and replacing `=` with `==` to comply with SymPy's `Eq()` format.
* **Parsing**: Splits the equation into left-hand side (lhs) and right-hand side (rhs).
* **Symbolic Parsing**: Converts the text into symbolic SymPy expressions.
* **Solving**: Uses `solve()` to compute the value(s) of the variable(s).

```python
except Exception as e:
    print("❌ Could not parse or solve the equation:", e)
```

* **Error Handling**: Catches and displays errors if OCR or parsing fails.

---

## 📊 Output/Result Meaning

* **Extracted Equation**: The textual representation of the equation recognized by OCR. Example: `2x+3=9`
* **Solution**: The solved value of the variable (e.g., `x=3`) derived using symbolic computation.

### Example

Given an image of a handwritten equation like:

```
 2x + 3 = 9
```

**Output**:

```
📝 Extracted Equation: 2x + 3 = 9
✅ Solution: [3]
```

**Meaning**: The system successfully interpreted the handwritten equation and solved for `x = 3`.

---

## 🔍 Limitations

* OCR accuracy highly depends on handwriting clarity and image quality.
* Complex or multi-variable equations may need more advanced parsing logic.

---

## 🛠 Requirements

* Python
* OpenCV
* pytesseract
* sympy
* Tesseract OCR installed and added to PATH

---

## 📌 Usage

1. Place a handwritten equation image as `equation.png`.
2. Run the script.
3. View the extracted equation and its solution in the console.

---

## ✅ Applications

* Educational tools
* Assistive tech for math learning
* Digitizing handwritten notes
