import streamlit as st
import torch
import torchvision.transforms as transforms
import os
from PIL import Image, ImageDraw
from datetime import datetime
from passport_utils import extract_passport


# ----------- Setup -----------
# Set class names here
class_names = ["ID Card", "Passport"]

# Load the trained model
@st.cache_resource
def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=None)
    model.fc = torch.nn.Linear(2048, len(class_names))  # match your output classes
    model.load_state_dict(torch.load("./best_model_passport3.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ----------- Preprocessing -----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------- Streamlit UI -----------
st.title("📄 ID Document Classifier App")
st.markdown("Upload a **passport** or **driver’s license** image and get classification with confidence score.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    # Preprocess and Predict
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_idx].item()

    pred_label = class_names[pred_idx]
    st.success(f"**Predicted: {pred_label}**")
    st.info(f"Confidence: `{confidence:.2%}`")

    # Optional: Draw Box (fake box for now — you can use YOLO later)
    if st.checkbox("🔲 Show Fake Bounding Box (top-right corner)"):
        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)
        w, h = draw_img.size
        box = [int(0.65 * w), int(0.05 * h), int(0.95 * w), int(0.35 * h)]
        draw.rectangle(box, outline="green", width=4)
        st.image(draw_img, caption="With Bounding Box", use_container_width=True)

    # Optional: Crop region
    if st.checkbox("✂️ Crop Top-Right Corner"):
        cropped_img = image.crop((int(0.65 * w), int(0.05 * h), int(0.95 * w), int(0.35 * h)))
        st.image(cropped_img, caption="🖼 Cropped Region", use_container_width=False)

    # Optional: Save result
    if st.button("📥 Download Prediction Result"):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = f"result_{now}.txt"
        with open(result_path, "w") as f:
            f.write(f"Prediction: {pred_label}\nConfidence: {confidence:.2%}")
        with open(result_path, "rb") as f:
            st.download_button("Download Result File", f, file_name=result_path)

def main():
    pass
    #extract_passport('./top_right_crop.jpg')
    #test('./video_right.mp4')

if __name__ == "__main__":
    main()
    
