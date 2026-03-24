import streamlit as st
st.set_page_config(page_title="Smart Attendance", layout="wide")

import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
from PIL import Image
import io

try:

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

    # ------------------ FACE MATCH ------------------
    def recognize_face(image_bytes):
        if not os.path.exists("dataset"):
            return "Unknown"

        try:
            uploaded_img = Image.open(io.BytesIO(image_bytes))
        except:
            return "Unknown"

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
            df = pd.read_csv(STUDENTS_FILE)
            if df.empty:
                return "Unknown"
            df["name"] = df["name"].str.lower()
            row = df[df["name"] == name.lower()]
            if not row.empty:
                return row.iloc[0]["class"]
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
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            df.to_csv(ATTENDANCE_FILE, index=False)
        else:
            pd.DataFrame([new_data]).to_csv(ATTENDANCE_FILE, index=False)

    # ------------------ UI ------------------

    st.sidebar.title("👩‍🏫 Teacher Panel")

    password = st.sidebar.text_input("Enter Password", type="password")

    if password == TEACHER_PASSWORD:
        st.sidebar.success("Access Granted")

    uploaded_file = st.file_uploader("Upload Student Image")

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()

        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, width=250)

        name = recognize_face(image_bytes)

        st.write("Detected:", name)

        class_name = get_student_class(name)

        mark_attendance(name, class_name, "neutral")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)

except Exception as e:
    st.error("APP CRASHEDD")
    st.exception(e)