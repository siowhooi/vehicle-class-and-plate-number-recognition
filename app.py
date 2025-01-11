import streamlit as st
import cv2
import easyocr
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import pytz

# Streamlit Title
st.title("Vehicle Classification and Plate Number Recognition")

# Create a left and right layout
col1, col2 = st.columns(2)

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

# Process uploaded image
if uploaded_image is not None:
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

            # Initialize EasyOCR reader
            reader = easyocr.Reader(['en'])

            # Separate vehicle and license plate detections
            vehicle_detections = []
            license_plate_detection = None

            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Debugging
                st.write(f"Class Detected: {class_name}, BBox: {x1}, {y1}, {x2}, {y2}")

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

                    # Debugging license plate detection
                    st.write(f"License Plate BBox: {x1}, {y1}, {x2}, {y2}")
                    license_plate_detection = {
                        "bbox": (x1, y1, x2, y2),
                        "image": image_rgb[y1:y2, x1:x2]
                    }

            # Time Formatting
            kl_timezone = pytz.timezone('Asia/Kuala_Lumpur')
            utc_time = datetime.now(pytz.utc)
            kl_time = utc_time.astimezone(kl_timezone)
            formatted_kl_time = kl_time.strftime("%d/%m/%Y %H:%M")

            # Handle License Plate OCR
            if license_plate_detection:
                plate_image = license_plate_detection["image"]
                
                # Preprocess Plate Image (Optional, depending on OCR results)
                plate_image_gray = cv2.cvtColor(plate_image, cv2.COLOR_RGB2GRAY)
                _, plate_image_thresh = cv2.threshold(plate_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                st.image(plate_image_thresh, caption="Preprocessed Plate Image for OCR")

                text_results = reader.readtext(plate_image_thresh, detail=0)
                st.write(f"OCR Results: {text_results}")
                recognized_text = ' '.join(text_results) if text_results else "Not Detected"
            else:
                recognized_text = "Not Detected"

            # Process vehicle and plate association
            for vehicle in vehicle_detections:
                st.session_state['results_data'].append({
                    "Datetime": formatted_kl_time,
                    "Vehicle Class": vehicle["class"],
                    "Vehicle Type": vehicle["vehicle_type"],
                    "Plate Number": recognized_text,
                })

            # Display image with detected bounding boxes
            with col1:
                plate_image_rgb = image.copy()
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(plate_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                st.image(cv2.cvtColor(plate_image_rgb, cv2.COLOR_BGR2RGB), caption="Detected Vehicle", use_container_width=True)

            # Results Table
            with col2:
                st.subheader("Results")
                if st.session_state['results_data']:
                    st.table(st.session_state['results_data'])
                else:
                    st.write("No data available yet.")

        except Exception as e:
            st.error(f"Error during inference: {e}")
