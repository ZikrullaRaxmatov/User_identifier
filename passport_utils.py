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

                # Extract text from preprocessed image
                text = pytesseract.image_to_string(img)

                # Example OCR result
                ocr_text = text.replace('\n', ' ').strip()

                # Extract Passport Number (e.g., FA6752048)
                passport_number = re.search(r'\b[A-Z]{2}\d{7}\b', ocr_text)

                if passport_number:
                    print("Passport No:", passport_number.group())
                    return passport_number.group(), img

                mrz_lines = re.findall(r'[A-Z0-9<]{40,}', ocr_text)
                
                if mrz_lines:
                    mrz_line = mrz_lines[0]
                    print("MRZ Line:", mrz_line)

                    match = re.search(r'[A-Z]{2}\d{7}', mrz_line)
                    if match:
                        print("Passport No:", match.group())
                        return match.group(), img
                    
            count_img += 1
            #cv2.imshow("Camera Test", img)

            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break
                
    #cap.release()
    #cv2.destroyAllWindows()
        
#extract_passport(2)