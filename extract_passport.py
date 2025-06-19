import re
import cv2 
import pytesseract


def get_extract_passport(img):

    if count_img % 5 ==0:
                
        #flipped = cv2.flip(img, -1)

        # Convert to grayscale for better OCR results
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Optional: thresholding or blurring
        gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

        # Extract text from preprocessed image
        text = pytesseract.image_to_string(gray)

        # Example OCR result
        ocr_text = text.replace('\n', ' ').strip()

        print(ocr_text)

        # Extract Passport Number (e.g., FA6752048)
        passport_number = re.search(r'\b[A-Z]{2}\d{7}\b', ocr_text)

        if passport_number:
            print("Passport No:", passport_number.group())

        # Step 1: Find all MRZ-like lines (at least 40 characters, all uppercase/number/<)
        mrz_lines = re.findall(r'[A-Z0-9<]{40,}', ocr_text)
        print(mrz_lines)
        print(len(mrz_lines))

        # Step 2: If second line exists, extract it and get passport number
        if mrz_lines:
            mrz_line = mrz_lines[0]
            print("MRZ Line:", mrz_line)

            # Step 3: Extract passport number (2 letters + 7 digits)
            match = re.search(r'[A-Z]{2}\d{7}', mrz_line)
            if match:
                print("Passport No:", match.group())
            else:
                print("Passport No not found in MRZ Line")
        else:
            print("MRZ Line not found.", count_img)
        
    count_img += 1