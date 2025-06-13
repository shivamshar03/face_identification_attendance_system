# 🧠 Face Identification & Attendance System

This is a Streamlit-based application that uses [DeepFace](https://github.com/serengil/deepface) to recognize faces either from a **camera** or **uploaded image** and mark attendance to a CSV file. You can also view attendance records from a separate page.

---

## 🚀 Features

- 📷 Take photos directly from your webcam  
- 🧑‍🔬 Face recognition using DeepFace and a custom image database  
- 📝 Attendance marking with name, date, and time  
- 📊 Live attendance viewer page  
- 🔐 No internet required after setup  

---

## 🗂️ Project Structure

face_attendance/  
├── .streamlit/  
│   └── config.toml               ← Optional: custom theme settings  
├── pages/  
│   └── 1_Attendance_Record.py    ← Page 2: View attendance records  
├── 0_Face_Recognition.py         ← Page 1: Upload/take photo & identify face  
├── known_faces/                  ← Face image database (one folder per person)  
│   ├── Elon_Musk/1.jpg  
│   └── ...  
├── attendance.csv                ← CSV file to store attendance records  
└── requirements.txt  

---

## ⚙️ Installation

1. Clone this repository or download the folder.  
2. Create a virtual environment:  
   - On Linux/macOS:  
     `python -m venv venv && source venv/bin/activate`  
   - On Windows:  
     `python -m venv venv && venv\Scripts\activate.bat`  

3. Install dependencies:  
   `pip install -r requirements.txt`  

4. Run the app:  
   `streamlit run 0_Face_Recognition.py`  

---

## 📁 Adding Known Faces

- Create subfolders inside `known_faces/`, each named after the person.
- Place one or more clear face images inside each folder.

**Example:**

known_faces/  
├── Elon_Musk/  
│   └── 1.jpg  
├── Taylor_Swift/  
│   └── 1.jpg  

---

## 📸 Camera Support

Streamlit's `st.camera_input()` is used for capturing images directly from webcam. This works only in a **web browser**, not in mobile or desktop Streamlit apps.

---

## 📊 Attendance Output

Attendance is recorded in a `attendance.csv` file in this format:

Name | Date | Time  
-----|------|------  
Elon_Musk | 2025-06-12 | 10:32:14  

---

## ✅ Requirements

- Python 3.8 to 3.10 recommended  
- Works offline after initial setup  

---

## 📦 Dependencies

All dependencies are listed in `requirements.txt`.

---

## 🛡 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

Built with ❤️ by **[Shivam Sharma]**
