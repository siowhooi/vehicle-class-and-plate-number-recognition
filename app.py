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

# Initialize results list to store data
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

# Toll rate data
fixed_toll_rates = {
    "Gombak Toll Plaza": {
        "Class 0": 0.00,
        "Class 1": 6.00,
        "Class 2": 12.00,
        "Class 3": 18.00,
        "Class 4": 3.00,
        "Class 5": 5.00,
    }
}

variable_toll_rates = {
    ("Jalan Duta, Kuala Lumpur", "Juru, Penang"): {
        "Class 0": 0.00,
        "Class 1": 35.51,
        "Class 2": 64.90,
        "Class 3": 86.50,
        "Class 4": 17.71,
        "Class 5": 21.15,
    },
    ("Seremban, Negeri Sembilan", "Jalan Duta, Kuala Lumpur"): {
        "Class 0": 0.00,
        "Class 1": 10.58,
        "Class 2": 19.50,
        "Class 3": 29.50,
        "Class 4": 5.33,
        "Class 5": 7.95,
    },
    ("Seremban, Negeri Sembilan", "Juru, Penang"): {
        "Class 0": 0.00,
        "Class 1": 43.95,
        "Class 2": 80.50,
        "Class 3": 107.20,
        "Class 4": 22.06,
        "Class 5": 30.95,
    },
}

# Store entry and exit history for variable toll plazas
vehicle_entries = {}

# Define function to calculate toll fare
def calculate_toll_fare(toll, mode, vehicle_class, plate_number):
    if toll == "Gombak Toll Plaza":
        return fixed_toll_rates[toll].get(vehicle_class, 0.00)
    elif mode == "Entry":
        vehicle_entries[plate_number] = toll
        return "-"
    elif mode == "Exit":
        entry_toll = vehicle_entries.pop(plate_number, None)
        if entry_toll:
            route = tuple(sorted([entry_toll, toll]))
            return variable_toll_rates.get(route, {}).get(vehicle_class, 0.00)
    return "-"

# Process uploaded image
if uploaded_image:
    # Load YOLO model
    model = YOLO(r"best.pt")  # Update with your YOLO weights path

    # Read and preprocess the image
    image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run YOLO detection
    results = model(image)

    # Initialize EasyOCR reader
    reader = easyocr.Reader(["en"])

    for result in results[0].boxes:
        class_id = int(result.cls)
        class_name = model.names[class_id]
        x1, y1, x2, y2 = map(int, result.xyxy[0])

        # Draw detections on image
        color = (0, 255, 0) if class_name not in ["license_plate", "license_plate_taxi"] else (255, 0, 0)
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_rgb, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Process vehicle detections
        if class_name not in ["license_plate", "license_plate_taxi"]:
            vehicle_class = vehicle_classes.get(class_name, "Unknown")

        # Process license plate detections
        if class_name in ["license_plate", "license_plate_taxi"]:
            plate_image = image_rgb[y1:y2, x1:x2]
            plate_text = reader.readtext(plate_image, detail=0)
            plate_number = "".join(plate_text).upper() if plate_text else "N/A"

            # Determine toll mode and calculate fare
            mode = "Entry Only" if toll_plaza == "Gombak Toll Plaza" else ("Entry" if plate_number not in vehicle_entries else "Exit")
            toll_fare = calculate_toll_fare(toll_plaza, mode, vehicle_class, plate_number)

            # Append results
            results_data.append(
                {
                    "Datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "Vehicle Class": vehicle_class,
                    "Plate Number": plate_number,
                    "Toll": toll_plaza,
                    "Mode": mode,
                    "Toll Fare (RM)": f"{toll_fare:.2f}" if toll_fare != "-" else "-",
                }
            )

    # Display image with detections
    with col1:
        st.image(image_rgb, caption="Detected Vehicles and License Plates", use_column_width=True)

    # Display results
    with col2:
        if results_data:
            st.subheader("Results")
            st.write("Detected vehicles and license plate information:")
            st.table(results_data)
