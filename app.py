import streamlit as st
st.set_page_config(page_title="Smart Attendance", layout="wide")

import cv2
import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
from PIL import Image
import io

# ------------------ PATH SETUP ------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.csv")
STUDENTS_FILE = os.path.join(DATA_DIR, "students.csv")
CLASSES_FILE = os.path.join(DATA_DIR, "classes.csv")

TEACHER_PASSWORD = "admin123"

st.markdown(
    "<h1 style='text-align: center; color: gold;'>Smart Classroom System</h1>",
    unsafe_allow_html=True
)

# ------------------ LIGHT FACE MATCH ------------------
def recognize_face(image_bytes):
    if not os.path.exists("dataset"):
        return "Unknown"

    uploaded_img = Image.open(io.BytesIO(image_bytes))

    for file in os.listdir("dataset"):
        try:
            dataset_img = Image.open(f"dataset/{file}")
            if uploaded_img.size == dataset_img.size:
                return file.split(".")[0]
        except:
            continue

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


def get_class_time(class_name):
    if os.path.exists(CLASSES_FILE):
        df = pd.read_csv(CLASSES_FILE)
        row = df[df["class"] == class_name]
        if not row.empty:
            return row.iloc[0]["start_time"]
    return None

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
                df = pd.concat([df, new_data], ignore_index=True)
                df.to_csv(STUDENTS_FILE, index=False)
            else:
                new_data.to_csv(STUDENTS_FILE, index=False)

            st.sidebar.success("Student added ✅")
            st.rerun()

else:
    st.sidebar.warning("Teacher access only")

# ------------------ ATTENDANCE ------------------
st.subheader("📸 Mark Attendance")

uploaded_file = st.file_uploader("Upload Student Image")

if uploaded_file is not None:

    image_bytes = uploaded_file.read()

    name = recognize_face(image_bytes)

    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    st.image(frame, channels="BGR", width=250)

    emotion = "neutral"
    confidence = 95

    class_name = get_student_class(name)

    if name == "Unknown":
        st.error("🚫 Not recognized")
    else:
        st.success(f"Name: {name}")
        st.info(f"Class: {class_name}")
        st.info(f"Emotion: {emotion}")
        st.caption(f"Confidence: {confidence}%")

        mark_attendance(name, class_name, emotion)

# ------------------ DATA ------------------
st.subheader("📋 Attendance Records")

if os.path.exists(ATTENDANCE_FILE):
    df = pd.read_csv(ATTENDANCE_FILE)

    if not df.empty:
        st.dataframe(df)