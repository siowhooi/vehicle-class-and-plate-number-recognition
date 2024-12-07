import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import datetime
from PIL import Image

# Load YOLO model
model = YOLO(r"runs/detect/train/weights/best.pt")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Streamlit app
st.set_page_config(layout="wide", page_title="Toll System")

# Layout
st.title("Smart Tolling System")
col1, col2 = st.columns(2)

# Left side
with col1:
    st.subheader("Input Section")

    # Dropdown for toll plaza selection
    toll_plaza = st.selectbox(
        "Select Toll Plaza:",
        ["Gombak Toll Plaza", "Jalan Duta, Kuala Lumpur", "Seremban, Negeri Sembilan", "Juru, Penang"]
    )

    # Upload image or use webcam
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])
    use_camera = st.button("Capture from Camera")

    if uploaded_file or use_camera:
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
        else:
            st.warning("Camera capture is not yet implemented.")
            img = None

        if img is not None:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Run YOLO inference
            results = model(img_rgb)

            # Draw detections
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                class_id = int(result.cls)
                class_name = model.names[class_id]

                color = (0, 255, 0) if class_name != 'license_plate' else (255, 0, 0)
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_rgb, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            st.image(img_rgb, caption="Detected Image", use_column_width=True)

            # Process for license plate
            vehicle_class = None
            plate_number = "Not Detected"
            for result in results[0].boxes:
                class_id = int(result.cls)
                class_name = model.names[class_id]

                if class_name == 'license_plate':
                    x1, y1, x2, y2 = map(int, result.xyxy[0])
                    plate_img = img_rgb[y1:y2, x1:x2]
                    plate_text = reader.readtext(plate_img, detail=0)

                    if plate_text:
                        plate_number = ''.join(plate_text)

                else:
                    vehicle_class = class_name

# Right side
with col2:
    st.subheader("Detection Results")
    datetime_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    toll_mode = "Closed Toll System" if toll_plaza != "Gombak Toll Plaza" else "Open Toll System"

    # Sample fare calculation
    fare = 5.00 if vehicle_class == "car" else 10.00 if vehicle_class == "truck" else 2.00

    # Display result
    st.table(
        [
            {
                "Datetime": datetime_now,
                "Vehicle Class": vehicle_class or "Not Detected",
                "Plate Number": plate_number,
                "Toll": toll_plaza,
                "Mode": toll_mode,
                "Toll Fare (RM)": f"{fare:.2f}"
            }
        ]
    )
