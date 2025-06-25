import os

def find_user_by_id(folder_path, target_id):
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

