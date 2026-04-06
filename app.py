import io
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Smart Attendance", layout="wide")

# ------------------ OPTIONAL FACE RECOGNITION ------------------
try:
    import face_recognition

    FACE_LIB_AVAILABLE = True
except Exception:
    FACE_LIB_AVAILABLE = False

# ------------------ PATH SETUP ------------------
DATA_DIR = "data"
DATASET_DIR = "dataset"

ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.csv")
STUDENTS_FILE = os.path.join(DATA_DIR, "students.csv")
CLASSES_FILE = os.path.join(DATA_DIR, "classes.csv")

TEACHER_PASSWORD = os.getenv("TEACHER_PASSWORD", "admin123")
DEFAULT_EMOTION = "neutral"
HASH_SIZE = 8
MAX_HASH_DISTANCE = 8
FACE_DISTANCE_THRESHOLD = 0.6
ALLOWED_IMAGE_TYPES = ["png", "jpg", "jpeg"]

os.makedirs(DATA_DIR, exist_ok=True)

# ------------------ HELPERS ------------------

def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def normalize_class(class_name: str) -> str:
    return " ".join(class_name.strip().upper().split())


def safe_read_csv(path: str, columns: list[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=columns)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns]


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def load_students() -> pd.DataFrame:
    df = safe_read_csv(STUDENTS_FILE, ["name", "class"])
    df["name"] = df["name"].astype(str).str.strip().str.lower()
    df["class"] = df["class"].astype(str).str.strip().str.upper()
    return df


def save_students(df: pd.DataFrame) -> None:
    save_csv(df, STUDENTS_FILE)


def load_classes() -> list[str]:
    if os.path.exists(CLASSES_FILE):
        df = safe_read_csv(CLASSES_FILE, ["class"])
        classes = df["class"].astype(str).str.strip().str.upper().tolist()
    else:
        classes = load_students()["class"].tolist()
    classes = [c for c in classes if c]
    return sorted(set(classes))


def save_classes(classes: list[str]) -> None:
    df = pd.DataFrame({"class": sorted(set(classes))})
    save_csv(df, CLASSES_FILE)


def refresh_classes_from_students() -> None:
    students = load_students()
    classes = students["class"].dropna().astype(str).tolist()
    save_classes(classes)


def image_ahash(image: Image.Image, hash_size: int = HASH_SIZE) -> int:
    img = image.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    pixels = np.asarray(img, dtype=np.float32)
    avg = float(pixels.mean())
    bits = (pixels > avg).astype(np.uint8).flatten()
    hash_value = 0
    for bit in bits:
        hash_value = (hash_value << 1) | int(bit)
    return hash_value


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def list_dataset_files() -> list[str]:
    if not os.path.exists(DATASET_DIR):
        return []
    files = []
    for file in os.listdir(DATASET_DIR):
        ext = os.path.splitext(file)[1].lower().lstrip(".")
        if ext in ALLOWED_IMAGE_TYPES:
            files.append(file)
    return files


def dataset_signature(dataset_files: list[str]) -> tuple[tuple[str, float], ...]:
    signature = []
    for file in dataset_files:
        path = os.path.join(DATASET_DIR, file)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        signature.append((file, mtime))
    return tuple(signature)


def encode_face(image: Image.Image) -> np.ndarray | None:
    if not FACE_LIB_AVAILABLE:
        return None
    try:
        array = np.array(image)
        locations = face_recognition.face_locations(array, model="hog")
        encodings = face_recognition.face_encodings(array, locations)
    except Exception:
        return None
    if not encodings:
        return None
    return encodings[0]


@st.cache_data(show_spinner=False)
def build_face_database(signature: tuple[tuple[str, float], ...]) -> tuple[list[str], np.ndarray]:
    if not FACE_LIB_AVAILABLE or not signature:
        return [], np.empty((0, 128))

    names: list[str] = []
    encodings: list[np.ndarray] = []

    for file, _mtime in signature:
        path = os.path.join(DATASET_DIR, file)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue

        encoding = encode_face(img)
        if encoding is None:
            continue

        names.append(os.path.splitext(file)[0])
        encodings.append(encoding)

    if not encodings:
        return [], np.empty((0, 128))

    return names, np.vstack(encodings)


# ------------------ FACE MATCHING ------------------

def recognize_face_real(image: Image.Image) -> str:
    encoding = encode_face(image)
    if encoding is None:
        return "Unknown"

    dataset_files = list_dataset_files()
    if not dataset_files:
        return "Unknown"

    signature = dataset_signature(dataset_files)
    names, known_encodings = build_face_database(signature)

    if not names:
        return "Unknown"

    distances = face_recognition.face_distance(known_encodings, encoding)
    best_index = int(np.argmin(distances))

    if distances[best_index] <= FACE_DISTANCE_THRESHOLD:
        return names[best_index]

    return "Unknown"


def recognize_face_fallback(image: Image.Image) -> str:
    dataset_files = list_dataset_files()
    if not dataset_files:
        return "Unknown"

    try:
        target_hash = image_ahash(image)
    except Exception:
        return "Unknown"

    best_name = "Unknown"
    best_distance = None

    for file in dataset_files:
        path = os.path.join(DATASET_DIR, file)
        try:
            dataset_img = Image.open(path).convert("RGB")
            candidate_hash = image_ahash(dataset_img)
            distance = hamming_distance(target_hash, candidate_hash)
        except Exception:
            continue

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_name = os.path.splitext(file)[0]

    if best_distance is not None and best_distance <= MAX_HASH_DISTANCE:
        return best_name

    return "Unknown"


def recognize_face(image: Image.Image) -> str:
    if FACE_LIB_AVAILABLE:
        return recognize_face_real(image)
    return recognize_face_fallback(image)


def get_student_class(name: str) -> str:
    df = load_students()
    row = df[df["name"] == normalize_name(name)]
    if not row.empty:
        return row.iloc[0]["class"]
    return "Unknown"


def mark_attendance(name: str, class_name: str, emotion: str) -> str:
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")

    new_data = {
        "Name": name,
        "Class": class_name,
        "Date": today,
        "Time": now.strftime("%I:%M:%S %p"),
        "Emotion": emotion,
    }

    df = safe_read_csv(ATTENDANCE_FILE, ["Name", "Class", "Date", "Time", "Emotion"])

    if not df.empty:
        if ((df["Name"] == name) & (df["Date"] == today)).any():
            return "already_marked"

    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    save_csv(df, ATTENDANCE_FILE)
    return "marked"


def delete_student(name: str, remove_attendance: bool) -> None:
    normalized = normalize_name(name)

    students = load_students()
    students = students[students["name"] != normalized]
    save_students(students)

    if os.path.exists(DATASET_DIR):
        for ext in ALLOWED_IMAGE_TYPES:
            path = os.path.join(DATASET_DIR, f"{normalized}.{ext}")
            if os.path.exists(path):
                os.remove(path)

    if remove_attendance:
        attendance = safe_read_csv(ATTENDANCE_FILE, ["Name", "Class", "Date", "Time", "Emotion"])
        if not attendance.empty:
            attendance = attendance[attendance["Name"] != normalized]
            save_csv(attendance, ATTENDANCE_FILE)

    refresh_classes_from_students()


def reset_attendance() -> None:
    empty = pd.DataFrame(columns=["Name", "Class", "Date", "Time", "Emotion"])
    save_csv(empty, ATTENDANCE_FILE)


# ------------------ TITLE ------------------
st.title("Smart Classroom System")
st.caption("Attendance tracking with optional face recognition")

if not FACE_LIB_AVAILABLE:
    st.info("Face recognition library not installed. Using fallback image matching.")


# ------------------ TEACHER PANEL ------------------
st.sidebar.title("Teacher Panel")
password = st.sidebar.text_input("Enter Password", type="password")

if password == TEACHER_PASSWORD:
    st.sidebar.success("Access granted")
    if TEACHER_PASSWORD == "admin123":
        st.sidebar.info("Tip: set TEACHER_PASSWORD to change the default password.")

    st.sidebar.subheader("Add Student")

    student_name = st.sidebar.text_input("Student Name")
    class_options = load_classes()
    class_choice = st.sidebar.selectbox("Class", ["Add new class"] + class_options)

    new_class = ""
    if class_choice == "Add new class":
        new_class = st.sidebar.text_input("New Class")

    student_image = st.sidebar.file_uploader(
        "Upload Image",
        type=ALLOWED_IMAGE_TYPES,
        key="teacher_upload",
    )

    if st.sidebar.button("Add/Update Student"):
        errors = []

        if not student_name.strip():
            errors.append("Student name is required.")

        selected_class = new_class if class_choice == "Add new class" else class_choice
        if not selected_class.strip():
            errors.append("Class is required.")

        if student_image is None:
            errors.append("Student image is required.")

        if errors:
            for error in errors:
                st.sidebar.error(error)
        else:
            normalized_name = normalize_name(student_name)
            class_name = normalize_class(selected_class)

            os.makedirs(DATASET_DIR, exist_ok=True)

            try:
                img = Image.open(student_image).convert("RGB")
            except Exception:
                st.sidebar.error("Invalid image file.")
            else:
                if FACE_LIB_AVAILABLE and encode_face(img) is None:
                    st.sidebar.error("No face detected in the uploaded image.")
                else:
                    img.save(os.path.join(DATASET_DIR, f"{normalized_name}.jpg"), format="JPEG")

                    students = load_students()
                    if normalized_name in students["name"].values:
                        students.loc[students["name"] == normalized_name, "class"] = class_name
                        action = "updated"
                    else:
                        students = pd.concat(
                            [students, pd.DataFrame([{"name": normalized_name, "class": class_name}])],
                            ignore_index=True,
                        )
                        action = "added"

                    save_students(students)

                    classes = load_classes()
                    if class_name not in classes:
                        classes.append(class_name)
                        save_classes(classes)

                    st.sidebar.success(f"Student {action} successfully.")
                    st.rerun()

    st.sidebar.subheader("Admin Tools")
    students_df = load_students()

    if students_df.empty:
        st.sidebar.info("No students available.")
    else:
        student_to_delete = st.sidebar.selectbox(
            "Delete Student",
            students_df["name"].tolist(),
            format_func=lambda value: value.title(),
        )
        delete_attendance = st.sidebar.checkbox(
            "Also remove their attendance records",
            value=True,
            key="delete_attendance",
        )

        if st.sidebar.button("Delete Student"):
            delete_student(student_to_delete, delete_attendance)
            st.sidebar.success("Student deleted.")
            st.rerun()

    st.sidebar.markdown("Clear all attendance records")
    reset_confirm = st.sidebar.text_input("Type RESET to confirm", key="reset_confirm")

    if st.sidebar.button("Reset Attendance"):
        if reset_confirm == "RESET":
            reset_attendance()
            st.sidebar.success("Attendance cleared.")
            st.rerun()
        else:
            st.sidebar.error("Please type RESET to confirm.")
else:
    st.sidebar.warning("Teacher access only")


# ------------------ ATTENDANCE ------------------
st.subheader("Mark Attendance")

input_mode = st.radio("Input source", ["Camera", "Upload"], horizontal=True)

image_bytes = None

if input_mode == "Camera":
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        image_bytes = camera_image.getvalue()
else:
    uploaded_file = st.file_uploader("Upload Student Image", type=ALLOWED_IMAGE_TYPES)
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()

if image_bytes:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        st.error("Invalid image file.")
    else:
        st.image(image, width=250)

        name = recognize_face(image)
        emotion = DEFAULT_EMOTION
        class_name = get_student_class(name)

        if name == "Unknown":
            st.error("Not recognized")
        else:
            st.success(f"Name: {name}")
            if class_name == "Unknown":
                st.warning("Class not found for this student.")
            else:
                st.info(f"Class: {class_name}")
            st.info(f"Emotion: {emotion}")

            status = mark_attendance(name, class_name, emotion)

            if status == "already_marked":
                st.warning("Already marked today")
            else:
                st.success("Attendance marked")


# ------------------ DATA ------------------
st.subheader("Class Attendance Dashboard")

students_df = load_students()
attendance_df = safe_read_csv(ATTENDANCE_FILE, ["Name", "Class", "Date", "Time", "Emotion"])
dashboard_date = st.date_input(
    "Dashboard date",
    value=datetime.now().date(),
    key="dashboard_date",
)

if students_df.empty:
    st.info("No students registered yet.")
else:
    classes = load_classes()
    if not classes:
        st.info("No classes available yet.")
    else:
        summary_rows = []
        selected_date = dashboard_date.strftime("%Y-%m-%d")

        for class_name in classes:
            total_students = students_df[students_df["class"] == class_name].shape[0]
            if total_students == 0:
                continue

            present_count = attendance_df[
                (attendance_df["Class"] == class_name)
                & (attendance_df["Date"] == selected_date)
            ]["Name"].dropna().nunique()

            attendance_percent = round((present_count / total_students) * 100, 1)

            summary_rows.append(
                {
                    "Class": class_name,
                    "Students": total_students,
                    "Present": present_count,
                    "Attendance %": attendance_percent,
                }
            )

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True)
            st.bar_chart(summary_df.set_index("Class")["Attendance %"])
        else:
            st.info("No class data to summarize yet.")

st.subheader("Attendance Records")

if attendance_df.empty:
    st.warning("No data yet")
else:
    branches = sorted(attendance_df["Class"].dropna().unique())

    if not branches:
        st.warning("No class data yet")
    else:
        tabs = st.tabs(branches)

        for i, branch in enumerate(branches):
            with tabs[i]:
                branch_df = attendance_df[attendance_df["Class"] == branch]

                st.markdown(f"### {branch} Attendance")
                st.dataframe(branch_df, use_container_width=True)

                st.download_button(
                    f"Download {branch} CSV",
                    branch_df.to_csv(index=False),
                    file_name=f"{branch}_attendance.csv",
                    mime="text/csv",
                )

                selected_date = st.date_input(
                    f"Select Date ({branch})",
                    value=datetime.now().date(),
                    key=f"{branch}_date",
                )
                filtered = branch_df[branch_df["Date"] == selected_date.strftime("%Y-%m-%d")]
                st.dataframe(filtered, use_container_width=True)

                st.subheader("Student Insights")
                students_in_branch = branch_df["Name"].dropna().unique().tolist()

                if students_in_branch:
                    student = st.selectbox(
                        f"Select Student ({branch})",
                        students_in_branch,
                        key=f"{branch}_student",
                    )

                    student_df = branch_df[branch_df["Name"] == student]

                    if not student_df.empty:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Classes", len(student_df))
                        col2.metric("Days", student_df["Date"].nunique())

                        emotion_counts = student_df["Emotion"].dropna().value_counts()
                        if not emotion_counts.empty:
                            mood = emotion_counts.idxmax().capitalize()
                            col3.metric("Mood", mood)
                            st.bar_chart(emotion_counts)
                        else:
                            col3.metric("Mood", "N/A")
                else:
                    st.info("No students found for this class yet.")

                st.subheader("Class Mood Today")
                today = datetime.now().strftime("%Y-%m-%d")
                today_df = branch_df[branch_df["Date"] == today]

                if not today_df.empty:
                    mood_counts = today_df["Emotion"].dropna().value_counts()

                    if not mood_counts.empty:
                        fig, ax = plt.subplots(figsize=(3, 3))
                        ax.pie(mood_counts, autopct="%1.0f%%")
                        st.pyplot(fig)
                    else:
                        st.info("No mood data for today yet.")
                else:
                    st.info("No attendance marked for today yet.")
