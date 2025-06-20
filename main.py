import re
import cv2
import pytesseract

from passport_utils import extract_passport

def main():
    print(extract_passport())
    

if __name__ == "__main__":
    main()