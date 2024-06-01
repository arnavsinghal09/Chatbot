from pdf2image import convert_from_path
import pytesseract

def pdf_to_text(pdf_path):
    images = convert_from_path(pdf_path)
    text = ''
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

print(pdf_to_text("./file0565.pdf"))