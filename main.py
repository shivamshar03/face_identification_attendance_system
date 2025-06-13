import streamlit as st
from deepface import DeepFace
from PIL import Image
import tempfile
import os
import pandas as pd
from datetime import datetime

st.title("📸 Face Identification with Camera")

# Camera input
camera_image = st.camera_input("Take a picture")

attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(attendance_file, index=False)

if camera_image is not None:
    st.image(camera_image, caption="Captured Image", use_column_width=True)

    # Save the image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(camera_image.getvalue())
        tmp_path = tmp.name

    try:
        st.info("🔍 Identifying face...")

        # Run DeepFace face search
        result = DeepFace.find(img_path=tmp_path, db_path="known_faces/", enforce_detection=True)

        if len(result[0]) > 0:
            matched_path = result[0].iloc[0]["identity"]
            person_name = os.path.basename(os.path.dirname(matched_path))
            st.success(f"✅ Identified: {person_name}")

            matched_image = Image.open(matched_path)
            st.image(matched_image, caption=f"Matched with: {person_name}", width=300)

            # Record attendance
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            df = pd.read_csv(attendance_file)
            if not ((df["Name"] == person_name) & (df["Date"] == date)).any():
                new_row = pd.DataFrame([[person_name, date, time]], columns=["Name", "Date", "Time"])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(attendance_file, index=False)
                st.success("📝 Attendance marked.")
            else:
                st.warning("🕒 Attendance already marked for today.")
        else:
            st.error("❌ No matching person found.")
    except Exception as e:
        st.error(f"Error: {e}")
