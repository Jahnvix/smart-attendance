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

# ------------------ CONFIG ------------------
TEACHER_PASSWORD = "admin123"

# ------------------ TITLE ------------------
st.markdown(
    "<h1 style='text-align: center; color: gold;'>Smart Classroom System</h1>",
    unsafe_allow_html=True
)

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
        try:
            df = pd.read_csv(ATTENDANCE_FILE)

            if ((df["Name"] == name) & (df["Date"] == today)).any():
                return "already_marked"

            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            df.to_csv(ATTENDANCE_FILE, index=False)
            return "marked"
        except:
            return "marked"
    else:
        pd.DataFrame([new_data]).to_csv(ATTENDANCE_FILE, index=False)
        return "marked"


def get_class_time(class_name):
    if os.path.exists(CLASSES_FILE):
        try:
            df = pd.read_csv(CLASSES_FILE)
            if df.empty:
                return None
            row = df[df["class"] == class_name]
            if not row.empty:
                return row.iloc[0]["start_time"]
        except:
            return None
    return None

# ------------------ TEACHER PANEL ------------------
st.sidebar.title("👩‍🏫 Teacher Panel")

password = st.sidebar.text_input("Enter Password", type="password")

if password == TEACHER_PASSWORD:
    st.sidebar.success("Access Granted")

    st.sidebar.subheader("Add Student")
    st.sidebar.info("System auto-refreshes after adding")

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
                try:
                    df = pd.read_csv(STUDENTS_FILE)

                    if student_name.lower() in df["name"].values:
                        st.sidebar.warning("Student already exists")
                    else:
                        df = pd.concat([df, new_data], ignore_index=True)
                        df.to_csv(STUDENTS_FILE, index=False)
                        st.sidebar.success("Student added ✅")
                        st.rerun()
                except:
                    new_data.to_csv(STUDENTS_FILE, index=False)
                    st.rerun()
            else:
                new_data.to_csv(STUDENTS_FILE, index=False)
                st.sidebar.success("Student added ✅")
                st.rerun()
        else:
            st.sidebar.warning("Fill all fields")

    # -------- Set Class Timing --------
    st.sidebar.subheader("⏰ Set Class Timing")

    class_name_input = st.sidebar.text_input("Class Name (e.g. CSE-A)")
    class_time = st.sidebar.time_input("Start Time")

    if st.sidebar.button("Save Class Timing"):
        if class_name_input:
            new_data = pd.DataFrame([{
                "class": class_name_input,
                "start_time": class_time.strftime("%H:%M")
            }])

            if os.path.exists(CLASSES_FILE):
                try:
                    df = pd.read_csv(CLASSES_FILE)
                except:
                    df = pd.DataFrame(columns=["class", "start_time"])

                df = df[df["class"] != class_name_input]
                df = pd.concat([df, new_data], ignore_index=True)
                df.to_csv(CLASSES_FILE, index=False)
            else:
                new_data.to_csv(CLASSES_FILE, index=False)

            st.sidebar.success("Class timing saved ✅")
            st.rerun()
        else:
            st.sidebar.warning("Enter class name")

else:
    st.sidebar.warning("Teacher access only")

# ------------------ SHOW CLASS TIMINGS ------------------
st.subheader("📚 Class Timings")

if os.path.exists(CLASSES_FILE):
    try:
        class_df = pd.read_csv(CLASSES_FILE)

        if not class_df.empty:
            st.dataframe(class_df)
        else:
            st.info("No class timings added yet")

    except:
        st.warning("Error reading class timings")
else:
    st.info("No class timings set yet")

# ------------------ STUDENT ATTENDANCE ------------------
st.subheader("📸 Mark Attendance")

uploaded_file = st.file_uploader("Upload Student Image")

if uploaded_file is not None:
    frame = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    if frame is not None:

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(frame, channels="BGR", width=250)

        try:
            result = DeepFace.find(
                img_path=frame,
                db_path="dataset",
                model_name="Facenet",
                enforce_detection=False
            )

            name = "Unknown"
            distance = 1

            if len(result) > 0 and len(result[0]) > 0:
                best = result[0].sort_values(by="distance").iloc[0]
                distance = best["distance"]

                if distance < 0.35:
                    name = os.path.basename(best["identity"]).split('.')[0]

        except:
            name = "Unknown"

        confidence = round((1 - distance) * 100, 2)

        try:
            emotion = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )[0]['dominant_emotion']
        except:
            emotion = "neutral"

        class_name = get_student_class(name)

        if name == "Unknown" or class_name == "Unknown":
            st.error("🚫 Not a part of class - Contact teacher")
        else:
            st.success(f"Name: {name}")
            st.info(f"Class: {class_name}")
            st.info(f"Emotion: {emotion}")
            st.caption(f"Confidence: {confidence}%")

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

        st.subheader("📅 Filter by Date")
        selected_date = st.date_input("Select Date")
        st.dataframe(df[df["Date"] == str(selected_date)])

        # 🔥 IMPROVED STUDENT INSIGHTS
        st.subheader("👤 Student Insights")

        student = st.selectbox("Select Student", df["Name"].unique())
        student_df = df[df["Name"] == student]

        if not student_df.empty:

            col1, col2, col3 = st.columns(3)
            col1.metric("Classes", len(student_df))
            col2.metric("Days", student_df["Date"].nunique())
            col3.metric("Mood", student_df["Emotion"].value_counts().idxmax().capitalize())

            st.markdown("---")

            col1, col2 = st.columns([1,1])

            with col1:
                st.markdown("**Mood Distribution**")
                st.bar_chart(student_df["Emotion"].value_counts())

            with col2:
                st.markdown("**Details**")
                st.write("Most common mood:",
                         student_df["Emotion"].value_counts().idxmax())
                st.write("Total records:", len(student_df))
                st.write("Last seen:",
                         student_df.iloc[-1]["Date"])

            st.markdown("**Recent Activity**")
            st.dataframe(student_df.tail(3), use_container_width=True)

        # ORIGINAL MOOD CHART (UNCHANGED)
        st.subheader("😊 Class Mood Today")

        today = datetime.now().strftime("%Y-%m-%d")
        today_df = df[df["Date"] == today]

        if not today_df.empty:
            mood_counts = today_df["Emotion"].value_counts()

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(2.5, 2.5))
                ax.pie(mood_counts, autopct='%1.0f%%', startangle=90, wedgeprops=dict(width=0.4))
                st.pyplot(fig)

            with col2:
                dominant = mood_counts.idxmax()
                st.success(f"Overall: {dominant.upper()}")

    else:
        st.warning("No data yet")
else:
    st.warning("No attendance file")