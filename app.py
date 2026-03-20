import streamlit as st
st.set_page_config(page_title="Smart Attendance", layout="wide")

import cv2
import pandas as pd
import numpy as np
from deepface.DeepFace import DeepFace
from datetime import datetime
import os
import matplotlib.pyplot as plt

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

.card-title {
    color: gold;
    font-size: 13px;
    letter-spacing: 1px;
}

.card-value {
    color: white;
    font-size: 20px;
    font-weight: bold;
}

/* -------- TITLE -------- */
.title {
    text-align: center;
    color: gold;
    animation: fadeIn 1s ease-in;
}

/* -------- BUTTONS -------- */
.stButton > button {
    background: linear-gradient(90deg, #000, gold);
    color: black;
    border-radius: 8px;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, gold, #000);
    color: white;
}

/* -------- SUCCESS PULSE -------- */
.success-box {
    padding: 10px;
    border-radius: 8px;
    background-color: #0f5132;
    color: white;
    animation: pulse 1.2s ease-in-out;
}

@keyframes pulse {
    0% { transform: scale(0.95); }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); }
}

/* -------- LOADING SPINNER -------- */
.loader {
    border: 4px solid #222;
    border-top: 4px solid gold;
    border-radius: 50%;
    width: 35px;
    height: 35px;
    animation: spin 1s linear infinite;
    margin: auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* -------- FADE -------- */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
.card-title {
    color: gold;
    font-size: 14px;
}
.card-value {
    color: white;
    font-size: 18px;
    font-weight: bold;
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

            # 🔥 CARDS UI
            col1, col2, col3 = st.columns(3)

            col1.markdown(f"""
            <div class="card">
                <div class="card-title">Name</div>
                <div class="card-value">{name}</div>
            </div>
            """, unsafe_allow_html=True)

            col2.markdown(f"""
            <div class="card">
                <div class="card-title">Class</div>
                <div class="card-value">{class_name}</div>
            </div>
            """, unsafe_allow_html=True)

            col3.markdown(f"""
            <div class="card">
                <div class="card-title">Emotion</div>
                <div class="card-value">{emotion}</div>
            </div>
            """, unsafe_allow_html=True)

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

        # ------------------ STUDENT INSIGHTS ------------------
        st.subheader("👤 Student Insights")

        student = st.selectbox("Select Student", df["Name"].unique())
        student_df = df[df["Name"] == student]

        if not student_df.empty:

            col1, col2, col3 = st.columns(3)

            col1.markdown(f"<div class='card'><div class='card-title'>Classes</div><div class='card-value'>{len(student_df)}</div></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='card'><div class='card-title'>Days</div><div class='card-value'>{student_df['Date'].nunique()}</div></div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='card'><div class='card-title'>Mood</div><div class='card-value'>{student_df['Emotion'].value_counts().idxmax().capitalize()}</div></div>", unsafe_allow_html=True)

            st.markdown("---")

            col1, col2 = st.columns([1,1])

            with col1:
                st.bar_chart(student_df["Emotion"].value_counts())

            with col2:
                st.write("Most common mood:", student_df["Emotion"].value_counts().idxmax())
                st.write("Total records:", len(student_df))
                st.write("Last seen:", student_df.iloc[-1]["Date"])

            st.dataframe(student_df.tail(3), use_container_width=True)

        # ------------------ MOOD ------------------
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