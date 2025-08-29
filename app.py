import os
import re
import cv2
import torch
import tempfile
import pytesseract
import pandas as pd
import streamlit as st
import torchvision.transforms as transforms
from user_utils import find_user_by_id
from PIL import Image

# ----------- Configurations -----------
st.set_page_config(page_title="ID Classifier App", layout="wide")

class_names = ["ID Card", "Passport"]

# Load model
@st.cache_resource
def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=None)
    model.fc = torch.nn.Linear(2048, len(class_names))
    model.load_state_dict(torch.load("./best_model.pt", map_location=torch.device('cpu')))
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
page = st.sidebar.radio("Go to", ["🏠 Main Page", "🧠 Identify User", "📂 All Users", "❔ Q&A"])


# -------------- Main Page --------------
if page == "🏠 Main Page":
    st.title("Welcome to the Document Classifier App")
    st.markdown("""
    This app allows you to:
    - Real time video or upload passport or ID videos/images.
    - Detect document type using a trained ResNet50 model.
    - View results and download classification.
    """)
    
    # Load an image (can be a file or URL)
    image = Image.open("uzpid.jpg")
    #image = image.resize((600, 500))

    # Display the image
    st.image(image, use_container_width=True)

# -------------- Identify User --------------
elif page == "🧠 Identify User":
    st.title("🧠 Identify Document Type")
    st.subheader("📷 Real-Time Video Streaming")
        
    # Create two columns
    col1, col2 = st.columns([2, 1])
    current_id = None
    current_img = None
    
    with col1:

        video_file = st.file_uploader("Upload a video or an image", type=["mp4", "mov", "avi", "jpg", "jpeg", "png"])
    
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

                        # Extract text from preprocessed image
                        text = pytesseract.image_to_string(img)

                        # Example OCR result
                        ocr_text = text.replace('\n', ' ').strip()

                        # Extract Passport Number (e.g., FA6752048)
                        passport_number = re.search(r'\b[A-Z]{2}\d{7}\b', ocr_text)

                        if passport_number:
                            #print("Passport No:", passport_number.group())
                            current_id = passport_number.group()
                            current_img = img
                            break

                        mrz_lines = re.findall(r'[A-Z0-9<]{40,}', ocr_text)
                        
                        if mrz_lines:
                            mrz_line = mrz_lines[0]
                            print("MRZ Line:", mrz_line)

                            match = re.search(r'[A-Z]{2}\d{7}', mrz_line)
                            if match:
                                #print("Passport No:", match.group())
                                current_id = match.group()
                                current_img = img
                                break
                        
                    count_img += 1
                    current_img = img
                    resized_img = cv2.resize(img, (450, 300))
                    stframe.image(resized_img)
                    
                if current_img is not None:
                    resized_img = cv2.resize(current_img, (450, 300))
                    stframe.image(resized_img)
                    
        
               
                        
    with col2:
        
        if current_img is not None:
            
            st.markdown("___")
            st.markdown("___")
            st.title("Results!!!")
            st.markdown("___")
            
            pil_img = Image.fromarray(current_img)

            img_tensor = transform(pil_img).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                print(output)
                pred_idx = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1)[0][pred_idx].item()
                print(confidence)
                
            
            pred_label = class_names[pred_idx]
            st.success(f"**Predicted: {pred_label}**")
            st.success(f"**ID: {current_id}**")
            st.info(f"Confidence: `{confidence:.2%}`")
                
        else:
            st.write("No Data")
            
    st.markdown("___")
    folder_path = './results/'
    if not os.path.exists(folder_path):
        st.error("❌ Folder not found.")
    else:
        user = find_user_by_id(folder_path, current_id)
        if user:
            st.success("✅ User Found!!!")
            df = pd.DataFrame([user])  # Single row DataFrame
            st.table(df)  #  Show result in table format
        else:
            st.warning("⚠️ User not found.")
            
    
# -------------- All Users Page --------------
elif page == "📂 All Users":
    st.title("📂 All Users")
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
elif page == "❔ Q&A":
    
    st.title("📬 Contact Information")
    st.markdown("""
    **Project Author**: Zikrulla Rakhmatov  

    If you have any questions or suggestions, feel free to reach out!
    """)
    st.text_area("💬 Leave a message:")
    st.button("Send Message")
    
# Spacer + Divider
st.sidebar.markdown("___")
st.sidebar.markdown("### 📬 Contact")

st.sidebar.markdown("""
📧 **Email**: zikrullarakhmatov@gmail.com  
☎️ **Tel**: +8210 8143 0080  
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

    
