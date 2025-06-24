import re
import cv2 
import pytesseract
import streamlit as st

def passport(path):

    cap = cv2.VideoCapture(path)
    frame_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

        # Optional stop condition
        if st.button("Stop", key='stop'):
            break

    cap.release()


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
