import streamlit as st
import torch
import torchvision.transforms as transforms
import os
from PIL import Image, ImageDraw
from datetime import datetime
from passport_utils import extract_passport, passport
from streamlit_webrtc import webrtc_streamer
from video_processor import VideoProcessor


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
    
    ctx = webrtc_streamer(
        key="ocr-app",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 250},
                "height": {"ideal": 200},
        }, 
        "audio": False
        }
    )

    print(ctx)
        

    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img_tensor = transform(image).unsqueeze(0)
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

        # Draw fake box
        if st.checkbox("🔲 Show Bounding Box"):
            draw_img = image.copy()
            draw = ImageDraw.Draw(draw_img)
            w, h = draw_img.size
            box = [int(0.65 * w), int(0.05 * h), int(0.95 * w), int(0.35 * h)]
            draw.rectangle(box, outline="green", width=4)
            st.image(draw_img, caption="With Bounding Box", use_container_width=True)

        # Crop ROI
        if st.checkbox("✂️ Crop Top-Right"):
            cropped = image.crop((int(0.65 * w), int(0.05 * h), int(0.95 * w), int(0.35 * h)))
            st.image(cropped, caption="Cropped Region", use_container_width=False)

        # Download result
        if st.button("📥 Download Prediction"):
            with open(result_file, "rb") as f:
                st.download_button("Download Result File", f, file_name=os.path.basename(result_file))

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


def main():
    pass
    #extract_passport('./top_right_crop.jpg')
    #test('./video_right.mp4')

if __name__ == "__main__":
    main()
    
