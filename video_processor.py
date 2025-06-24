import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import re
import cv2
import pytesseract

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_text = ""

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 🔄 1. Convert to NumPy
        img = frame.to_ndarray(format="bgr24")

        # 🧪 2. Preprocessing
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Optionally threshold or denoise for better OCR
        #gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]

        # 🔍 3. OCR: read text from frame
        text = pytesseract.image_to_string(img)
        
        # Example OCR result
        ocr_text = text.replace('\n', ' ').strip()

        print(ocr_text)

        # Extract Passport Number (e.g., FA6752048)
        passport_number = re.search(r'\b[A-Z]{2}\d{7}\b', ocr_text)

        if passport_number:
            print("Passport No:", passport_number.group())
            return passport_number.group(), frame

        mrz_lines = re.findall(r'[A-Z0-9<]{40,}', ocr_text)

        if mrz_lines:
            mrz_line = mrz_lines[0]
            print("MRZ Line:", mrz_line)

            match = re.search(r'[A-Z]{2}\d{7}', mrz_line)
            if match:
                print("Passport No:", match.group())
                return match.group(), frame
            else:
                print("Passport No not found in MRZ Line")
        else:
            print("MRZ Line not found.")
        
        #self.latest_text = text

        # 🖼 5. Return the processed frame
        #return av.VideoFrame.from_ndarray(img, format="bgr24")
