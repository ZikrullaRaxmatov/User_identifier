import re
import cv2
import torch
import pytesseract
import streamlit as st
from torchvision import transforms

class_names = ["ID Card", "Passport"]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model
@st.cache_resource
def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=None)
    model.fc = torch.nn.Linear(2048, len(class_names))
    model.load_state_dict(torch.load("./best_model.pt", map_location=torch.device('cpu')))
    model.eval()
    return model


def predict_img(model, transform, img):
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_idx].item()

    pred_label = class_names[pred_idx]
    print(f"**Predicted: {pred_label}**")
    print(f"Confidence: `{confidence:.2%}`")


def extract_text(path):
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
                
                print(ocr_text)

                # Extract Passport Number (e.g., FA6752048)
                passport_number = re.search(r'\b[A-Z]{2}\d{7}\b', ocr_text)

                if passport_number:
                    print("Passport No:", passport_number.group())
                    return passport_number.group(), img

                mrz_lines = re.findall(r'[A-Z0-9<]{40,}', ocr_text)
                
                if mrz_lines:
                    mrz_line = mrz_lines[0]

                    match = re.search(r'[A-Z]{2}\d{7}', mrz_line)
                    if match:
                        print("Passport No:", match.group())
                        return match.group(), img
                    
            count_img += 1
                
            #resized = cv2.resize(img, (540, 500))
            cv2.imshow("Camera Test", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()


extract_text(2)