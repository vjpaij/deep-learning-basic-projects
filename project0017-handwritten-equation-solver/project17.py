import cv2
import pytesseract
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
 
# Initialize symbolic math
x, y, z = symbols('x y z')
 
# Load image of handwritten equation
image = cv2.imread("equation.png")  # Replace with your own image path
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# Optional: Thresholding for better OCR performance
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, 11, 2)
 
# OCR with pytesseract
custom_config = r'--oem 3 --psm 6'
extracted_text = pytesseract.image_to_string(gray, config=custom_config)
 
print("📝 Extracted Equation:", extracted_text)
 
try:
    # Clean up the expression (basic sanitization)
    equation = extracted_text.strip().replace(" ", "").replace("=", "==")
 
    # Parse and solve using SymPy
    lhs, rhs = equation.split("==")
    expr = Eq(parse_expr(lhs), parse_expr(rhs))
    solution = solve(expr)
 
    print("✅ Solution:", solution)
 
except Exception as e:
    print("❌ Could not parse or solve the equation:", e)