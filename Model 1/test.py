from pdf2image import convert_from_path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '"C:\\Programs\\TesseractOCR\\tesseract.exe"'

def pdf_to_text(pdf_path):
    images = convert_from_path(pdf_path,500,poppler_path=r"C:\\Programs\\poppler-24.02.0\\Library\bin")
    text = ''
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

print(pdf_to_text("C:\\Study\\SEM2\\ML\\Chatbot\\Chatbot\\test.pdf"))
