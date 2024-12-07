import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from datetime import datetime

# Load YOLO model and EasyOCR
model = YOLO(r"best.pt")
reader = easyocr.Reader(['en'])

# Toll fare data
fixed_toll_rates = {
    "Class 1": 6.00, "Class 2": 12.00, "Class 3": 18.00,
    "Class 4": 3.00, "Class 5": 5.00, "Class 0": 0.00
}

variable_toll_rates = {
    ("Jalan Duta, Kuala Lumpur", "Juru, Penang"): {
        "Class 1": 35.51, "Class 2": 64.90, "Class 3": 86.50,
        "Class 4": 17.71, "Class 5": 21.15, "Class 0": 0.00
    },
    ("Seremban, Negeri Sembilan", "Jalan Duta, Kuala Lumpur"): {
        "Class 1": 10.58, "Class 2": 19.50, "Class 3": 29.50,
        "Class 4": 5.33, "Class 5": 7.95, "Class 0": 0.00
    },
    ("Seremban, Negeri Sembilan", "Juru, Penang"): {
        "Class 1": 43.95, "Class 2": 80.50, "Class 3": 107.20,
        "Class 4": 22.06, "Class 5": 30.95, "Class 0": 0.00
    }
}

# Result log
results_log = []

# Streamlit layout
st.set_page_config(layout="wide")
st.sidebar.title("Toll Plaza Selection")

# Sidebar inputs
location = st.sidebar.selectbox(
    "Select Toll Plaza",
    ["Gombak Toll Plaza", "Jalan Duta, Kuala Lumpur",
     "Seremban, Negeri Sembilan", "Juru, Penang"]
)

uploaded_file = st.sidebar.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png"])
capture_camera = st.sidebar.button("Capture from Camera")

# Helper functions
def detect_vehicle(image):
    results = model(image)
    vehicle_info = []
    for result in results[0].boxes:
        class_id = int(result.cls)
        vehicle_class = model.names[class_id]
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        vehicle_info.append({
            "class": vehicle_class,
            "bbox": (x1, y1, x2, y2)
        })
    return vehicle_info

def recognize_plate(image):
    text = reader.readtext(image, detail=0)
    return "".join(text)

def calculate_toll(entry, exit, vehicle_class):
    if entry == "Gombak Toll Plaza":
        return fixed_toll_rates.get(vehicle_class, 0.00)
    key = (entry, exit) if (entry, exit) in variable_toll_rates else (exit, entry)
    return variable_toll_rates.get(key, {}).get(vehicle_class, 0.00)

# Right panel display
col1, col2 = st.columns(2)

with col1:
    st.header("Detection Input")
    if uploaded_file or capture_camera:
        # Load image
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
        else:
            # Mocked camera capture (replace with actual camera capture logic)
            st.write("Camera feature not yet implemented.")
            img = None

        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vehicle_info = detect_vehicle(img_rgb)
            for v_info in vehicle_info:
                x1, y1, x2, y2 = v_info["bbox"]
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_rgb, v_info["class"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            st.image(img_rgb, caption="Processed Image with Bounding Boxes", use_column_width=True)

            # Assume vehicle detection includes class info
            for v_info in vehicle_info:
                vehicle_class = v_info["class"]
                if vehicle_class.startswith("class"):
                    plate_crop = img_rgb[v_info["bbox"][1]:v_info["bbox"][3], v_info["bbox"][0]:v_info["bbox"][2]]
                    plate_number = recognize_plate(plate_crop)
                    current_time = datetime.now().strftime("%d/%m/%Y %H:%M")
                    toll_fare = calculate_toll(location, location, vehicle_class.split("_")[1])

                    # Append to log
                    results_log.append({
                        "Datetime": current_time,
                        "Vehicle Class": vehicle_class.split("_")[1].capitalize(),
                        "Plate Number": plate_number,
                        "Toll Mode": "Entry Only" if toll_fare == 0 else "Exit",
                        "Toll Fare (RM)": toll_fare
                    })

with col2:
    st.header("Results")
    st.table(results_log)
