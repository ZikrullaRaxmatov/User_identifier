import os
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import tempfile
import cv2
import pytesseract
import re

# ----------- Configurations -----------
st.set_page_config(page_title="ID Classifier App", layout="wide")

# Class names — update these based on your model
class_names = ["ID Card", "Passport"]

# Load model
@st.cache_resource
def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=None)
    model.fc = torch.nn.Linear(2048, len(class_names))
    model.load_state_dict(torch.load("./best_model_passport3.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------- Sidebar --------------
st.sidebar.title("🔍 Navigation")

# -------------- Contact Page --------------
page = st.sidebar.radio("Go to", ["🏠 Main Page", "🧠 Identify User", "📂 All Users", "📬 Contact"])


# -------------- Main Page --------------
if page == "🏠 Main Page":
    st.title("Welcome to the Document Classifier App")
    st.markdown("""
    This app allows you to:
    - Real time video or upload passport or ID images.
    - Detect document type using a trained ResNet50 model.
    - View results and download classification.
    """)

# -------------- Identify User --------------
elif page == "🧠 Identify User":
    st.title("🧠 Identify Document Type")
    st.subheader("📷 Real-Time Video Streaming")
        
    # Create two columns
    col1, col2 = st.columns([2, 1])
    current_id = None
    current_img = None
    
    with col1:
        video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        
        if video_file is not None:
            # ⏳ Save video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            
            stframe = st.empty()
            
            cap = cv2.VideoCapture(tfile.name)
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
                        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        # Optional: thresholding or blurring
                        #gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

                        # Extract text from preprocessed image
                        text = pytesseract.image_to_string(img)

                        # Example OCR result
                        ocr_text = text.replace('\n', ' ').strip()

                        #print(ocr_text)

                        # Extract Passport Number (e.g., FA6752048)
                        passport_number = re.search(r'\b[A-Z]{2}\d{7}\b', ocr_text)

                        if passport_number:
                            print("Passport No:", passport_number.group())
                            current_id, current_img = passport_number.group(), img
                            break

                        mrz_lines = re.findall(r'[A-Z0-9<]{40,}', ocr_text)
                        #print(mrz_lines)
                        #print(len(mrz_lines))

                        if mrz_lines:
                            mrz_line = mrz_lines[0]
                            print("MRZ Line:", mrz_line)

                            match = re.search(r'[A-Z]{2}\d{7}', mrz_line)
                            if match:
                                print("Passport No:", match.group())
                                current_id, current_img = match.group(), img
                                break
                            else:
                                print("Passport No not found in MRZ Line")
                        else:
                            print("MRZ Line not found.", count_img)
                        
                    count_img += 1
                    
                    resized_img = cv2.resize(img, (500, 450))
                    stframe.image(resized_img)

            
    with col2:
        
        if current_id:
            st.write("Passport ID: ", current_id)
            pil_img = Image.fromarray(img)

            img_tensor = transform(pil_img).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1)[0][pred_idx].item()

            pred_label = class_names[pred_idx]
            st.success(f"**Predicted: {pred_label}**")
            st.info(f"Confidence: `{confidence:.2%}`")

            # Save to results folder (for All Users)
            os.makedirs("results", exist_ok=True)
            result_file = f"results/{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(result_file, "w") as f:
                f.write(f"Prediction: {pred_label}\nConfidence: {confidence:.2%}")

            # Download result
            if st.button("📥 Download Prediction"):
                with open(result_file, "rb") as f:
                    st.download_button("Download Result File", f, file_name=os.path.basename(result_file))
        else:
            st.write("No Data")


        
# -------------- All Users Page --------------
elif page == "📂 All Users":
    st.title("📂 All Prediction Results")
    results_path = "results"
    if os.path.exists(results_path):
        files = os.listdir(results_path)
        if not files:
            st.warning("No results found yet.")
        else:
            for file in sorted(files, reverse=True):
                with open(os.path.join(results_path, file), "r") as f:
                    content = f.read()
                st.code(content, language="text")
    else:
        st.warning("No results directory found.")

# -------------- Contact Page --------------
elif page == "❔❕ Questions or Suggestions":
    st.title("📬 Contact Information")
    st.markdown("""
    **Project Author**: Zikrulla Rakhmatov  

    If you have any questions or suggestions, feel free to reach out!
    """)
    st.text_area("💬 Leave a message:")
    
# Spacer + Divider
st.sidebar.markdown("___")
st.sidebar.markdown("### 📬 Contact")

st.sidebar.markdown("""
📧 **Email**: zikrullarakhmatov@gmail.com  
☎️ **Tel**: +998 99 334 77 88  
""")

st.sidebar.markdown("""
<div style='display: flex; justify-content: center; gap: 20px;'>
    <a href="https://t.me/@ZikrullaRakhmatov" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/2111/2111646.png" width="28" title="Telegram">
    </a>
    <a href="https://youtube.com/@ZikrullaRakhmatov" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/1384/1384060.png" width="28" title="YouTube">
    </a>
    <a href="https://github.com/ZikrullaRaxmatov" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="28" title="GitHub">
    </a>
</div>
""", unsafe_allow_html=True)

    
