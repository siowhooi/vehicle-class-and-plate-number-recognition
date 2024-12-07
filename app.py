import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
from datetime import datetime
import pandas as pd

# Load YOLO model and EasyOCR
model = YOLO(r"C:\Users\wohen\PycharmProjects\test\runs\detect\train\weights\best.pt")
reader = easyocr.Reader(['en'])

# Mapping for vehicle classes and toll rates
vehicle_classes = {
    "class0_emergencyVehicle": ("Class 0", 0.00),
    "class1_lightVehicle": ("Class 1", 6.00),
    "class2_mediumVehicle": ("Class 2", 12.00),
    "class3_heavyVehicle": ("Class 3", 18.00),
    "class4_taxi": ("Class 4", 3.00),
    "class5_bus": ("Class 5", 5.00),
}

# Toll fare mappings
closed_toll_fares = {
    ("Jalan Duta, Kuala Lumpur", "Juru, Penang"): [35.51, 64.90, 86.50, 17.71, 21.15],
    ("Seremban, Negeri Sembilan", "Jalan Duta, Kuala Lumpur"): [10.58, 19.50, 29.50, 5.33, 7.95],
    ("Seremban, Negeri Sembilan", "Juru, Penang"): [43.95, 80.50, 107.20, 22.06, 30.95],
}

# Initialize a placeholder for toll log
if "toll_log" not in st.session_state:
    st.session_state.toll_log = []

# Function to calculate toll fare
def calculate_toll(entry, exit, vehicle_class):
    if vehicle_class == "class0_emergencyVehicle":
        return 0.00
    if entry == "Gombak Toll Plaza":
        return vehicle_classes[vehicle_class][1]
    for (start, end), rates in closed_toll_fares.items():
        if (entry, exit) in [(start, end), (end, start)]:
            class_idx = int(vehicle_class.split('_')[1][0]) - 1
            return rates[class_idx]
    return None

# Streamlit App Layout
st.set_page_config(layout="wide")
col1, col2 = st.columns(2)

# Left side (input)
with col1:
    st.header("Toll Detection")
    location = st.selectbox(
        "Choose Location",
        ["Gombak Toll Plaza", "Jalan Duta, Kuala Lumpur", "Seremban, Negeri Sembilan", "Juru, Penang"]
    )

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    capture_image = st.button("Capture Image from Camera")

    if uploaded_file or capture_image:
        # Load image
        image_path = uploaded_file if uploaded_file else "camera_image.jpg"  # Placeholder for camera integration
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR) if uploaded_file else None
        
        # Run YOLO detection
        results = model(image)
        detections = []
        for result in results[0].boxes:
            class_id = int(result.cls)
            class_name = model.names[class_id]
            if class_name.startswith("class"):
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                plate_crop = image[y1:y2, x1:x2]
                plate_text = "".join(reader.readtext(plate_crop, detail=0))
                detections.append({"class": class_name, "plate": plate_text, "location": location})

        # Update toll log
        for detection in detections:
            existing_entry = next(
                (log for log in st.session_state.toll_log if log["plate"] == detection["plate"] and log["location"] == detection["location"]), 
                None
            )
            mode = "Entry" if not existing_entry else "Exit"
            toll_fare = "-" if mode == "Entry" else calculate_toll(existing_entry["location"], detection["location"], detection["class"])
            st.session_state.toll_log.append({
                "datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "vehicle_class": vehicle_classes[detection["class"]][0],
                "plate": detection["plate"],
                "location": detection["location"],
                "mode": mode,
                "toll_fare": toll_fare,
            })

# Right side (output)
with col2:
    st.header("Toll Log")
    df = pd.DataFrame(st.session_state.toll_log)
    st.dataframe(df)
