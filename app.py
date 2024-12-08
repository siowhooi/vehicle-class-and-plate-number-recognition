import streamlit as st
import cv2
import easyocr
from ultralytics import YOLO
import numpy as np
from datetime import datetime

# Streamlit Title
st.title("Vehicle and License Plate Recognition")

# Initialize session state for vehicle entries
if 'vehicle_entries' not in st.session_state:
    st.session_state['vehicle_entries'] = {}

# Fixed toll rates for Gombak Toll Plaza
fixed_toll_rates = {
    "Class 0": 0.00,
    "Class 1": 6.00,
    "Class 2": 12.00,
    "Class 3": 18.00,
    "Class 4": 3.00,
    "Class 5": 5.00,
}

# Variable toll rates for other routes
variable_toll_rates = {
    ("Jalan Duta, Kuala Lumpur", "Juru, Penang"): {
        "Class 1": 35.51,
        "Class 2": 64.90,
        "Class 3": 86.50,
        "Class 4": 17.71,
        "Class 5": 21.15,
    },
    ("Seremban, Negeri Sembilan", "Jalan Duta, Kuala Lumpur"): {
        "Class 1": 10.58,
        "Class 2": 19.50,
        "Class 3": 29.50,
        "Class 4": 5.33,
        "Class 5": 7.95,
    },
    ("Seremban, Negeri Sembilan", "Juru, Penang"): {
        "Class 1": 43.95,
        "Class 2": 80.50,
        "Class 3": 107.20,
        "Class 4": 22.06,
        "Class 5": 30.95,
    },
}

# Vehicle classes
vehicle_classes = {
    "class0_emergencyVehicle": "Class 0",
    "class1_lightVehicle": "Class 1",
    "class2_mediumVehicle": "Class 2",
    "class3_heavyVehicle": "Class 3",
    "class4_taxi": "Class 4",
    "class5_bus": "Class 5",
}

# Create layout
col1, col2 = st.columns(2)

# Toll plaza selection and image upload
with col1:
    st.subheader("Detection")
    toll_plaza = st.selectbox(
        "Select Toll Plaza",
        [
            "Gombak Toll Plaza",
            "Jalan Duta, Kuala Lumpur",
            "Seremban, Negeri Sembilan",
            "Juru, Penang",
        ],
    )
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Functions
def process_image(image, model, reader):
    """Process image and detect vehicles and plates."""
    results_data = []
    results = model(image)  # YOLO inference

    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]

        # Skip if not a vehicle class
        if class_name not in vehicle_classes:
            continue

        # Vehicle class and bounding box
        vehicle_class = vehicle_classes[class_name]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, vehicle_class, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

        # License plate OCR
        plate_image = image[y1:y2, x1:x2]
        plate_text = reader.readtext(plate_image, detail=0)
        recognized_text = ''.join(plate_text).upper().replace(" ", "").strip() if plate_text else "N/A"

        # Determine Entry or Exit
        if recognized_text not in st.session_state['vehicle_entries']:
            mode = "Entry"
            st.session_state['vehicle_entries'][recognized_text] = {"plaza": toll_plaza, "class": vehicle_class}
            toll_fare = "-"
        else:
            mode = "Exit"
            entry_data = st.session_state['vehicle_entries'].pop(recognized_text)
            entry_plaza, entry_class = entry_data["plaza"], entry_data["class"]

            # Calculate toll
            route_key = tuple(sorted([entry_plaza, toll_plaza]))
            toll_fare = variable_toll_rates.get(route_key, {}).get(entry_class, 0.00)
            if entry_plaza == "Gombak Toll Plaza":
                toll_fare = fixed_toll_rates.get(entry_class, 0.00)

        results_data.append({
            "Datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "Vehicle Class": vehicle_class,
            "Plate Number": recognized_text,
            "Toll": toll_plaza,
            "Mode": mode,
            "Toll Fare (RM)": f"{toll_fare:.2f}" if toll_fare != "-" else "-",
        })

    return image, results_data

# Process uploaded image
if uploaded_image is not None:
    try:
        model = YOLO(r"best.pt")  # Load YOLO model
        reader = easyocr.Reader(['en'])  # Initialize EasyOCR

        # Read and process image
        image_data = np.frombuffer(uploaded_image.read(), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Invalid image. Please upload a valid image.")
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_image, results_data = process_image(image_rgb, model, reader)

            # Display processed image
            st.image(processed_image, caption="Detected Vehicles", use_column_width=True)

            # Display results
            with col2:
                st.subheader("Results")
                if results_data:
                    st.table(results_data)
                else:
                    st.write("No vehicles detected.")

    except Exception as e:
        st.error(f"Error during processing: {e}")
