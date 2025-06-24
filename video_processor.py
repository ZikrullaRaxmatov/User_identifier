import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import re
import cv2
import pytesseract

class VideoProcessor(VideoProcessorBase):
    
    def __init__(self):
        self.passport_number = None
        self.current_frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        
        # 🔄 1. Convert to NumPy
        img = frame.to_ndarray(format="bgr24")

        # 🧪 2. Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Optionally threshold or denoise for better OCR
        gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]

        # 🔍 3. OCR: read text from frame
        text = pytesseract.image_to_string(img)
        
        # Example OCR result
        ocr_text = text.replace('\n', ' ').strip()

        print(ocr_text)

        # Extract Passport Number
        passport_number_match = re.search(r'\b[A-Z]{2}\d{7}\b', ocr_text)
        passport_number = None

        if passport_number_match:
            passport_number = passport_number_match.group()
        else:
            mrz_lines = re.findall(r'[A-Z0-9<]{40,}', ocr_text)
            if mrz_lines:
                match = re.search(r'[A-Z]{2}\d{7}', mrz_lines[0])
                if match:
                    passport_number = match.group()

        if passport_number:
            # ✅ Store into session_state
            st.session_state["passport_number"] = passport_number
            st.session_state["latest_frame"] = img

        # 🖼 5. Return the processed frame
        return passport_number
        