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

    # Read the uploaded image
    image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference on the image
    results = model(image)

    # Initialize EasyOCR reader for license plate recognition
    reader = easyocr.Reader(['en'])

    # Process YOLO detections
    for result in results[0].boxes:
        class_id = int(result.cls)
        class_name = model.names[class_id]
        x1, y1, x2, y2 = map(int, result.xyxy[0])

        color = (0, 255, 0) if class_name not in ["license_plate", "license_plate_taxi"] else (255, 0, 0)
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_rgb, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # For license plates, crop and recognize text
        if class_name in ["license_plate", "license_plate_taxi"]:
            plate_image = image_rgb[y1:y2, x1:x2]
            plate_text = reader.readtext(plate_image, detail=0)

            # Recognized plate text
            plate_number = ''.join(plate_text).upper() if plate_text else "N/A"

            # Determine Mode based on toll plaza
            mode = "Entry Only" if toll_plaza == "Gombak Toll Plaza" else "Entry or Exit"

            # Add result to results_data
            vehicle_class = vehicle_classes.get(class_name, "Unknown")
            results_data.append(
                {
                    "Datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "Vehicle Class": vehicle_class,
                    "Plate Number": plate_number,
                    "Toll": toll_plaza,
                    "Mode": mode,
                }
            )

    # Display the image with detections in col1
    with col1:
        st.image(image_rgb, caption="Detected Vehicles and License Plates", use_column_width=True)

# Display results table in col2
with col2:
    if results_data:
        st.subheader("Results")
        st.write("Detected vehicles and license plate information:")
        st.table(results_data)
