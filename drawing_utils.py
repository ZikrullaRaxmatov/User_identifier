import cv2

# Load the image
image = cv2.imread("./id_raxmatova.jpg")

# Get image dimensions
height, width, _ = image.shape

# Define crop box (top-right corner)
# e.g., top 25% of height, right 25% of width
crop_height = int(height * 0.25)
crop_width = int(width * 0.25)

# Define region: top rows, right columns
cropped = image[0:crop_height, width - crop_width:width]

# Save or show
cv2.imwrite("top_right_crop.jpg", cropped)
print("✅ Saved cropped region as 'top_right_crop.jpg'")
