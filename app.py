import streamlit as st
import cv2
import easyocr
from ultralytics import YOLO
import numpy as np
from datetime import datetime

# Streamlit Title
st.title("Vehicle Classification and Plate Number Recognition")

# Create a left and right layout
col1, col2 = st.columns(2)

# Initialize a dictionary to track vehicle entry and exit
if 'vehicle_entries' not in st.session_state:
    st.session_state['vehicle_entries'] = {}

# Initialize results_data in session state if not already present
if 'results_data' not in st.session_state:
    st.session_state['results_data'] = []

# Vehicle classes and their types
vehicle_classes = {
    "class0_emergencyVehicle": ("Class 0", "Emergency Vehicle"),
    "class1_lightVehicle": ("Class 1", "Light Vehicle"),
    "class2_mediumVehicle": ("Class 2", "Medium Vehicle"),
    "class3_heavyVehicle": ("Class 3", "Heavy Vehicle"),
    "class4_taxi": ("Class 4", "Taxi"),
    "class5_bus": ("Class 5", "Bus"),
}

# Define image upload
with col1:
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Initialize EasyOCR reader with custom allowed characters
allowed_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
reader = easyocr.Reader(['en'], gpu=False, lang_list=['en'], char_list=allowed_characters)

# Process uploaded image
if uploaded_image is not None:
    # Check if the model file exists and load it
    try:
        model = YOLO(r"best.pt")  # Ensure the model path is correct
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()  # Stop execution if model loading fails

    # Read and decode the uploaded image
    image_data = uploaded_image.read()
    image = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Failed to decode image. Please try again with a valid image.")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            # Run YOLO inference
            results = model(image_rgb)  # Use the RGB image for inference

            # Separate vehicle and license plate detections
            vehicle_detections = []
            license_plate_detection = None

            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Skip unrelated detections
                if class_name not in vehicle_classes and class_name != "license_plate":
                    continue

                if class_name in vehicle_classes:
                    # Add vehicle detection
                    vehicle_class, vehicle_type = vehicle_classes[class_name]
                    vehicle_detections.append({
                        "class": vehicle_class,
                        "vehicle_type": vehicle_type,
                        "bbox": (x1, y1, x2, y2)
                    })

                elif class_name == "license_plate":
                    # Save license plate detection
                    h, w, _ = image_rgb.shape
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                    license_plate_detection = {
                        "bbox": (x1, y1, x2, y2),
                        "image": image_rgb[y1:y2, x1:x2]
                    }

            # Match license plates with vehicle detections
            if license_plate_detection:
                plate_image = license_plate_detection["image"]

                if plate_image.size > 0:
                    # Perform OCR with restricted character set
                    text_results = reader.readtext(plate_image, detail=0)
                    # Join text results and filter out unwanted characters
                    recognized_text = ''.join(text_results) if text_results else "Not Detected"
                else:
                    recognized_text = "Not Detected"

                # Match the license plate with the nearest vehicle detection
                license_plate_bbox = license_plate_detection["bbox"]
                matched_vehicle = None
                min_distance = float('inf')

                for vehicle in vehicle_detections:
                    vehicle_bbox = vehicle["bbox"]

                    # Calculate distance between bounding boxes
                    vehicle_center = ((vehicle_bbox[0] + vehicle_bbox[2]) / 2, (vehicle_bbox[1] + vehicle_bbox[3]) / 2)
                    license_plate_center = ((license_plate_bbox[0] + license_plate_bbox[2]) / 2, (license_plate_bbox[1] + license_plate_bbox[3]) / 2)
                    distance = np.sqrt((vehicle_center[0] - license_plate_center[0])**2 + (vehicle_center[1] - license_plate_center[1])**2)

                    if distance < min_distance:
                        min_distance = distance
                        matched_vehicle = vehicle

                # Append matched results
                if matched_vehicle:
                    st.session_state['results_data'].append({
                        "Datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
                        "Vehicle Class": matched_vehicle["class"],
                        "Vehicle Type": matched_vehicle["vehicle_type"],
                        "Plate Number": recognized_text,
                    })

            else:
                # Append vehicles without plates
                for vehicle in vehicle_detections:
                    st.session_state['results_data'].append({
                        "Datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
                        "Vehicle Class": vehicle["class"],
                        "Vehicle Type": vehicle["vehicle_type"],
                        "Plate Number": "Not Detected",
                    })

            # Display the image with YOLO detections
            with col1:
                plate_image_rgb = image.copy()
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(plate_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

                st.image(cv2.cvtColor(plate_image_rgb, cv2.COLOR_BGR2RGB), caption="Detected Vehicle", use_container_width=True)

            # Display the cropped plate image with recognized text
            with col2:
                st.subheader("License Plate Detection")
                if license_plate_detection and recognized_text != "Not Detected":
                    st.image(license_plate_detection["image"], caption=f"Detected License Plate: {recognized_text}", use_container_width=True)
                else:
                    st.write("No license plate detected.")

            # Display results in table format
            with col2:
                st.subheader("Results")

                # Display persistent results from session state
                if st.session_state['results_data']:
                    st.table(st.session_state['results_data'])
                else:
                    st.write("No data available yet.")
        except Exception as e:
            st.error(f"Error during inference: {e}")
