# Smart Attendance System

A Streamlit app that records attendance from student images. It supports live webcam capture, optional real face recognition, and per-class attendance insights.

---

## Features

- Webcam capture and image upload
- Optional real face recognition (fallback matcher included)
- Teacher panel to add or update students
- Admin tools to delete students and reset attendance
- Per-class attendance percentage dashboard
- Per-class tabs, date filtering, and CSV export
- Student insights and class mood charts

---

## Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- Pillow
- Matplotlib

---

## Project Structure

```
smart-attendance-system/
|
|-- app.py
|-- dataset/          # student images
|-- data/             # csv files
|-- requirements.txt
|-- README.md
```

---

## Installation and Setup

### 1. Clone Repository

```
git clone https://github.com/your-username/smart-attendance-system.git
cd smart-attendance-system
```

### 2. Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate   (Windows)
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. (Optional) Enable Real Face Recognition

```
pip install face-recognition
```

Note: `face-recognition` may require C++ build tools on Windows.

### 5. Run Application

```
streamlit run app.py
```

---

## Configuration

- `TEACHER_PASSWORD`: set this environment variable to change the teacher password. Default is `admin123`.

---

## Notes

- If `face-recognition` is not installed, the app uses a lightweight fallback matcher.
- Emotion is currently stored as `neutral` (placeholder).

---

## Future Enhancements

- Emotion detection integration
- Cloud deployment or mobile support

---

## Author

Jahnvi Gupta
