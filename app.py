import streamlit as st
import cv2
import easyocr
from ultralytics import YOLO
import numpy as np
from datetime import datetime

# Streamlit Title
st.title("Vehicle and License Plate Recognition")

# Create a left and right layout
col1, col2 = st.columns(2)

with col1:
    # Dropdown menu for toll plaza selection
    toll_plaza = st.selectbox(
        "Select Toll Plaza",
        [
            "Gombak Toll Plaza",
            "Jalan Duta, Kuala Lumpur",
            "Seremban, Negeri Sembilan",
            "Juru, Penang",
        ],
    )

    # File uploader for image upload
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Initialize a results list to store data
results_data = []

# Define vehicle classes
vehicle_classes = {
    "class0_emergencyVehicle": "Class 0",
    "class1_lightVehicle": "Class 1",
    "class2_mediumVehicle": "Class 2",
    "class3_heavyVehicle": "Class 3",
    "class4_taxi": "Class 4",
    "class5_bus": "Class 5",
}

if uploaded_image is not None:
    # Load the YOLO model
    model = YOLO(r"best.pt")  # Update with the path to your YOLO model weights

    # Read and decode the uploaded image
    image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run YOLO inference
    results = model(image)

    # Initialize EasyOCR reader for license plate recognition
    reader = easyocr.Reader(['en'])

    # Process YOLO detections
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw bounding box and label
        color = (0, 255, 0) if class_name not in ["license_plate", "license_plate_taxi"] else (255, 0, 0)
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_rgb, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # License plate OCR
        if class_name in ["license_plate", "license_plate_taxi"]:
            plate_image = image_rgb[y1:y2, x1:x2]
            plate_text = reader.readtext(plate_image, detail=0)
            recognized_text = ''.join(plate_text).upper() if plate_text else "N/A"

            # Store license plate information
            results_data.append(
                {
                    "Datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "Vehicle Class": "N/A",
                    "Plate Number": recognized_text,
                    "Toll": toll_plaza,
                    "Mode": "N/A",
                    "Toll Fare (RM)": "-",
                }
            )
        else:
            # Process vehicle class information
            vehicle_class = vehicle_classes.get(class_name, "Unknown")
            results_data.append(
                {
                    "Datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "Vehicle Class": vehicle_class,
                    "Plate Number": "N/A",
                    "Toll": toll_plaza,
                    "Mode": "N/A",
                    "Toll Fare (RM)": "-",
                }
            )

    # Display the image with YOLO detections
    with col1:
        st.image(image_rgb, caption="Detected Vehicles and License Plates", use_column_width=True)

    # Display results in table format
    with col2:
        st.subheader("Results")
        if results_data:
            st.table(results_data)
        else:
            st.write("No vehicles or license plates detected.")
