import streamlit as st
st.set_page_config(page_title="Smart Attendance", layout="wide")

import cv2
import pandas as pd
import numpy as np
from deepface import DeepFace
from datetime import datetime
import os
import matplotlib.pyplot as plt

# ------------------ PATH SETUP ------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.csv")
STUDENTS_FILE = os.path.join(DATA_DIR, "students.csv")
CLASSES_FILE = os.path.join(DATA_DIR, "classes.csv")

TEACHER_PASSWORD = "admin123"

# ------------------ TITLE ------------------
st.markdown(
    "<h1 style='text-align: center; color: gold;'>Smart Classroom System</h1>",
    unsafe_allow_html=True
)

# ------------------ FUNCTIONS ------------------

def get_student_class(name):
    if os.path.exists(STUDENTS_FILE):
        df = pd.read_csv(STUDENTS_FILE)
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

    st.sidebar.subheader("Add Student")

    student_name = st.sidebar.text_input("Student Name")
    student_class = st.sidebar.text_input("Class")
    student_image = st.sidebar.file_uploader("Upload Image", key="teacher_upload")

    if st.sidebar.button("Add Student"):
        if student_name and student_class and student_image:
            os.makedirs("dataset", exist_ok=True)

            with open(f"dataset/{student_name.lower()}.jpg", "wb") as f:
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
    frame = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    if frame is not None:
        st.image(frame, channels="BGR", width=250)

        try:
            result = DeepFace.find(img_path=frame, db_path="dataset", enforce_detection=False)

            name = "Unknown"
            if len(result) > 0 and len(result[0]) > 0:
                best = result[0].sort_values(by="distance").iloc[0]
                if best["distance"] < 0.35:
                    name = os.path.basename(best["identity"]).split('.')[0]
        except:
            name = "Unknown"

        emotion = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)[0]['dominant_emotion']
        class_name = get_student_class(name)

        if name == "Unknown":
            st.error("🚫 Not a part of class")
        else:
            st.success(f"Name: {name}")
            st.info(f"Class: {class_name}")
            st.info(f"Emotion: {emotion}")

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

        # 🔥 CREATE TABS PER BRANCH
        branches = sorted(df["Class"].dropna().unique())

        if branches:
            tabs = st.tabs(branches)

            for i, branch in enumerate(branches):
                with tabs[i]:

                    branch_df = df[df["Class"] == branch]

                    st.markdown(f"### 📘 {branch} Attendance")

                    st.dataframe(branch_df)

                    # Download per branch
                    st.download_button(
                        f"Download {branch} CSV",
                        branch_df.to_csv(index=False),
                        file_name=f"{branch}_attendance.csv"
                    )

                    # 📅 Date filter
                    selected_date = st.date_input(f"Select Date ({branch})", key=branch)
                    st.dataframe(branch_df[branch_df["Date"] == str(selected_date)])

                    # 👤 Student Insights
                    st.subheader("👤 Student Insights")

                    student = st.selectbox(
                        f"Select Student ({branch})",
                        branch_df["Name"].unique(),
                        key=f"{branch}_student"
                    )

                    student_df = branch_df[branch_df["Name"] == student]

                    if not student_df.empty:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Classes", len(student_df))
                        col2.metric("Days", student_df["Date"].nunique())
                        col3.metric("Mood", student_df["Emotion"].value_counts().idxmax().capitalize())

                        st.bar_chart(student_df["Emotion"].value_counts())

                    # 😊 Mood
                    st.subheader("😊 Class Mood Today")

                    today = datetime.now().strftime("%Y-%m-%d")
                    today_df = branch_df[branch_df["Date"] == today]

                    if not today_df.empty:
                        mood_counts = today_df["Emotion"].value_counts()

                        fig, ax = plt.subplots(figsize=(2.5, 2.5))
                        ax.pie(mood_counts, autopct='%1.0f%%')
                        st.pyplot(fig)

    else:
        st.warning("No data yet")
else:
    st.warning("No attendance file")