# User Identifier System

This project is a user identification system that captures a document using a camera, classifies it, extracts text information via OCR, and verifies it against a database. If a match is found, the system retrieves and reports all related user information.

The main goal is to enable real-time user verification with just a camera input.
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/9e830d16-88b2-466e-a223-e6f14edff383" />
⚠ This system works only uzbeks passports and ID cards now. But you can costumize to your document type



## 🚀 Features
- 📸 Camera Capture – Capture and process documents in real time
- 🧠 Deep Learning (ResNet50) – Document classification
- 🔎 OCR Integration – Extracts key text data from documents
- 🗄️ Database Check – Verifies user identity and retrieves stored information
- 🌐 Streamlit Frontend – Simple and interactive web-based UI
- ⚡ Python Backend – Efficient data processing and reporting
  

## 🛠️ Tech Stack
- Frontend: Streamlit
- Backend: Python
- Deep Learning Model: ResNet50
- OCR Engine: Tesseract OCR
- Database: Handmade DB

## 📥 Installation
1. Clone the repository
<pre> git clone https://github.com/ZikrullaRaxmatov/User_identifier.git  </pre>
<pre> cd your-repo-name </pre>

3. Create and activate a virtual environment (recommended)
<pre> python -m venv venv  </pre>
- source venv/bin/activate   # For Linux/Mac
- venv\Scripts\activate      # For Windows

5. Install dependencies
<pre> pip install -r requirements.txt  </pre>

7. Run the application
<pre> streamlit run main.py  </pre>

## 📖 Usage
- Open the app in your browser (Streamlit provides a local URL).
- Use your camera to scan or upload a document.
- The system will:
* Classify the type of document
* Extract text information using OCR
* Check against the database
* Display all user information if found

## 🤝 Contributing
Contributions are welcome! Please fork the repo and submit a pull request.
Steps to contribute:
1. Fork the project
2. Create a new branch (git checkout -b feature-xyz)
3. Commit your changes (git commit -m 'Add new feature')
4. Push to your branch (git push origin feature-xyz)
5. Open a Pull Request

## 📜 License
This project is licensed under the MIT License.

