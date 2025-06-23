import os
from PIL import Image
from torchvision import transforms

# Input & output directories
input_dir = "id" # Folder with original images
output_dir = "augmented_images"    # Folder for augmented images
os.makedirs(output_dir, exist_ok=True)

# Define augmentation pipeline
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

# For converting tensor back to image
to_pil = transforms.ToPILImage()

# Loop over all images
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_dir, filename)
        img = Image.open(image_path)

        base_name = os.path.splitext(filename)[0]
        
        for i in range(10):  # Create 10 augmented versions
            augmented = transform(img)
            augmented_img = to_pil(augmented)
            save_path = os.path.join(output_dir, f"{base_name}_aug_{i}.jpg")
            augmented_img.save(save_path)
            print(f"✅ Saved: {save_path}")
