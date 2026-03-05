import sys
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

try:
    pdfmetrics.registerFont(TTFont('Helvetica', 'C:/Windows/Fonts/arial.ttf'))
    pdfmetrics.registerFont(TTFont('Helvetica-Bold', 'C:/Windows/Fonts/arialbd.ttf'))
    pdfmetrics.registerFont(TTFont('Helvetica-Oblique', 'C:/Windows/Fonts/ariali.ttf'))
    print("Fonts registered successfully.")
except Exception as e:
    print(f"Error: {e}")
