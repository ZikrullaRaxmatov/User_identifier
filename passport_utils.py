import re
import cv2 
import pytesseract



def extract_passport(path):

    cap = cv2.VideoCapture(path)
    count_img = 0

    if not cap.isOpened():
        print("Camera not accessible or not found.")
    else:
        print("Camera is working! Press 'q' to quit.")
        while True:
            ret, img = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            if count_img % 5 == 0:
                        
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
                    #return passport_number.group()

                mrz_lines = re.findall(r'[A-Z0-9<]{40,}', ocr_text)
                #print(mrz_lines)
                #print(len(mrz_lines))

                if mrz_lines:
                    mrz_line = mrz_lines[0]
                    print("MRZ Line:", mrz_line)

                    match = re.search(r'[A-Z]{2}\d{7}', mrz_line)
                    if match:
                        print("Passport No:", match.group())
                        #return match.group()
                    else:
                        print("Passport No not found in MRZ Line")
                else:
                    print("MRZ Line not found.", count_img)
                
            count_img += 1
                
            cv2.imshow("Camera Test", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
    
    
def test(path):
    
    cap = cv2.VideoCapture(path)
    count_img = 0

    if not cap.isOpened():
        print("Camera not accessible or not found.")
    else:
        print("Camera is working! Press 'q' to quit.")
        while True:
            ret, img = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            if count_img % 5 == 0:

                # Load passport image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Optional: resize or denoise if OCR is weak
                gray = cv2.resize(gray, None, fx=1.5, fy=1.5)

                # Run OCR
                text = pytesseract.image_to_string(gray)
                print("===== RAW OCR TEXT =====")
                print(text)

                # Now extract structured fields
                info = {}

                # Extract Passport Number (AB1234567)
                passport_match = re.search(r'\b[A-Z]{2}\d{7}\b', text)
                if passport_match:
                    info['Passport Number'] = passport_match.group()

                # Extract Surname (assume word after 'FAMILYASI' or 'SURNAME')
                surname_match = re.search(r'FAMILIYASI\s*/\s*SURNAME\s*\n([A-Z]+)', text, re.IGNORECASE)
                if surname_match:
                    info['Surname'] = surname_match.group(1)

                # Extract Given Name (ISMI)
                given_name = re.search(r'ISMI\s*/\s*GIVEN\s+NAMES?\s*\n([A-Z]+)', text, re.IGNORECASE)
                if given_name:
                    info['Given Name'] = given_name.group(1)

                # Extract Nationality
                nat = re.search(r'FUQAROLIGI\s*/\s*NATIONALITY\s*\n([A-Z]+)', text, re.IGNORECASE)
                if nat:
                    info['Nationality'] = nat.group(1)

                # Extract Date of Birth
                dob = re.search(r'TUG\'ILGAN SANASI\s*/\s*DATE OF BIRTH\s*\n([\d\s]+)', text, re.IGNORECASE)
                if dob:
                    info['Date of Birth'] = dob.group(1).strip()

                # Extract Place of Birth
                pob = re.search(r'TUG\'ILGAN JOYI\s*/\s*PLACE OF BIRTH\s*\n([A-Z\s]+)', text, re.IGNORECASE)
                if pob:
                    info['Place of Birth'] = pob.group(1).strip()

                # Extract Date of Issue
                doi = re.search(r'BERILGAN SANASI\s*/\s*DATE OF ISSUE\s*\n([\d\s]+)', text, re.IGNORECASE)
                if doi:
                    info['Date of Issue'] = doi.group(1).strip()

                # Extract Expiry Date
                exp = re.search(r'AMAL QILISH MUDDATI\s*/\s*DATE OF EXPIRY\s*\n([\d\s]+)', text, re.IGNORECASE)
                if exp:
                    info['Date of Expiry'] = exp.group(1).strip()

                # Show structured result
                print("\n===== EXTRACTED INFORMATION =====")
                for key, value in info.items():
                    print(f"{key}: {value}")
            
            count_img += 1
            
            cv2.imshow("Camera Test", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
