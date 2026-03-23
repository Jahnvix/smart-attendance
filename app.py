import streamlit as st
st.set_page_config(page_title="Smart Attendance", layout="wide")

import cv2
import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import face_recognition   # ✅ NEW (lightweight)

# ------------------ CARD STYLE ------------------
st.markdown("""
<style>
.card {
    background: linear-gradient(145deg, #111, #1a1a1a);
    padding: 14px;
    border-radius: 12px;
    border: 1px solid #333;
    box-shadow: 0 0 10px rgba(255,215,0,0.15);
    margin-bottom: 12px;
    transition: all 0.3s ease-in-out;
    animation: fadeIn 0.6s ease-in;
}
.card:hover {
    transform: translateY(-6px) scale(1.03);
    box-shadow: 0 0 20px rgba(255,215,0,0.4);
    border: 1px solid gold;
}
.card-title { color: gold; font-size: 14px; }
.card-value { color: white; font-size: 18px; font-weight: bold; }

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ------------------ PATH SETUP ------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.csv")
STUDENTS_FILE = os.path.join(DATA_DIR, "students.csv")
CLASSES_FILE = os.path.join(DATA_DIR, "classes.csv")

TEACHER_PASSWORD = "admin123"

# ------------------ TITLE ------------------
st.markdown("<h1 style='text-align: center; color: gold;'>Smart Classroom System</h1>", unsafe_allow_html=True)

# ------------------ FACE RECOGNITION ------------------
def recognize_face(frame):
    known_faces = []
    known_names = []

    if not os.path.exists("dataset"):
        return "Unknown"

    for file in os.listdir("dataset"):
        try:
            img = face_recognition.load_image_file(f"dataset/{file}")
            enc = face_recognition.face_encodings(img)

            if len(enc) > 0:
                known_faces.append(enc[0])
                known_names.append(file.split(".")[0])
        except:
            continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)

    if len(encodings) > 0:
        matches = face_recognition.compare_faces(known_faces, encodings[0])
        if True in matches:
            return known_names[matches.index(True)]

    return "Unknown"

# ------------------ FUNCTIONS ------------------

def get_student_class(name):
    if os.path.exists(STUDENTS_FILE):
        try:
            df = pd.read_csv(STUDENTS_FILE)
            df["name"] = df["name"].str.lower()
            row = df[df["name"] == name.lower()]
            if not row.empty:
                return row.iloc[0]["class"]
        except:
            pass
    return "Unknown"


def mark_attendance(name, class_name, emotion):
    today = datetime.now().strftime("%Y-%m-%d")

    new_data = {
        "Name": name,
        "Class": class_name,
        "Date": today,
        "Time": datetime.now().strftime("%I:%M:%S %p"),
        "Emotion": emotion
    }

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)

        if ((df["Name"] == name) & (df["Date"] == today)).any():
            return "already_marked"

        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        return "marked"
    else:
        pd.DataFrame([new_data]).to_csv(ATTENDANCE_FILE, index=False)
        return "marked"


# ------------------ TEACHER PANEL ------------------
st.sidebar.title("👩‍🏫 Teacher Panel")
password = st.sidebar.text_input("Enter Password", type="password")

if password == TEACHER_PASSWORD:
    st.sidebar.success("Access Granted")

    student_name = st.sidebar.text_input("Student Name")
    student_class = st.sidebar.text_input("Class")
    student_image = st.sidebar.file_uploader("Upload Image", key="teacher_upload")

    if st.sidebar.button("Add Student"):
        if student_name and student_class and student_image:
            os.makedirs("dataset", exist_ok=True)

            path = f"dataset/{student_name.lower()}.jpg"
            with open(path, "wb") as f:
                f.write(student_image.read())

            new_data = pd.DataFrame([{
                "name": student_name.lower(),
                "class": student_class
            }])

            if os.path.exists(STUDENTS_FILE):
                df = pd.read_csv(STUDENTS_FILE)

                if student_name.lower() in df["name"].values:
                    st.sidebar.warning("Student already exists")
                else:
                    df = pd.concat([df, new_data], ignore_index=True)
                    df.to_csv(STUDENTS_FILE, index=False)
                    st.sidebar.success("Student added ✅")
                    st.rerun()
            else:
                new_data.to_csv(STUDENTS_FILE, index=False)
                st.sidebar.success("Student added ✅")
                st.rerun()

# ------------------ ATTENDANCE ------------------
st.subheader("📸 Mark Attendance")
uploaded_file = st.file_uploader("Upload Student Image")

if uploaded_file is not None:
    frame = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    if frame is not None:
        st.image(frame, channels="BGR", width=250)

        name = recognize_face(frame)
        emotion = "neutral"  # simplified

        class_name = get_student_class(name)

        if name == "Unknown" or class_name == "Unknown":
            st.error("🚫 Not a part of class - Contact teacher")
        else:
            col1, col2, col3 = st.columns(3)

            col1.markdown(f"<div class='card'><div class='card-title'>Name</div><div class='card-value'>{name}</div></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='card'><div class='card-title'>Class</div><div class='card-value'>{class_name}</div></div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='card'><div class='card-title'>Emotion</div><div class='card-value'>{emotion}</div></div>", unsafe_allow_html=True)

            status = mark_attendance(name, class_name, emotion)

            if status == "already_marked":
                st.warning("⚠️ Already marked present today")
            else:
                st.success("✅ Attendance marked")

# ------------------ DATA ------------------
st.subheader("📋 Attendance Records")

if os.path.exists(ATTENDANCE_FILE):
    df = pd.read_csv(ATTENDANCE_FILE)

    if not df.empty:
        st.dataframe(df)

        st.download_button("📥 Download CSV", df.to_csv(index=False), "attendance.csv")

        st.subheader("👤 Student Insights")
        student = st.selectbox("Select Student", df["Name"].unique())
        student_df = df[df["Name"] == student]

        if not student_df.empty:
            col1, col2, col3 = st.columns(3)

            col1.markdown(f"<div class='card'><div class='card-title'>Classes</div><div class='card-value'>{len(student_df)}</div></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='card'><div class='card-title'>Days</div><div class='card-value'>{student_df['Date'].nunique()}</div></div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='card'><div class='card-title'>Mood</div><div class='card-value'>{student_df['Emotion'].value_counts().idxmax().capitalize()}</div></div>", unsafe_allow_html=True)

            st.bar_chart(student_df["Emotion"].value_counts())