import os


def find_user_by_id(folder_path, target_id):
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                user_data = {}
                for line in file:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        user_data[key.strip()] = value.strip()
                if user_data.get("ID") == target_id:
                    return user_data
    return None


def find_user_by_id2(folder_path, target_id):
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith("ID:"):
                        user_id = line.strip().split("ID:")[1].strip()
                        if user_id == target_id:
                            # If found, return full content
                            return ''.join(lines)
    return "❌ User not found."



'''
# Save to results folder (for All Users)
os.makedirs("results", exist_ok=True)
result_file = f"results/{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(result_file, "w") as f:
    f.write(f"Prediction: {pred_label}\nConfidence: {confidence:.2%}")

# Download result
if st.button("📥 Download Prediction"):
    with open(result_file, "rb") as f:
        st.download_button("Download Result File", f, file_name=os.path.basename(result_file))
'''