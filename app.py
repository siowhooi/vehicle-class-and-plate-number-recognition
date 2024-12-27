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

# Vehicle classes
vehicle_classes = {
    "class0_emergencyVehicle": "Class 0",
    "class1_lightVehicle": "Class 1",
    "class2_mediumVehicle": "Class 2",
    "class3_heavyVehicle": "Class 3",
    "class4_taxi": "Class 4",
    "class5_bus": "Class 5",
}

# Define image upload
with col1:
    st.subheader("Detection")
  
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

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

            # Initialize EasyOCR reader
            reader = easyocr.Reader(['en'])

            # Process YOLO detections
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Skip non-vehicle detections
                if class_name not in vehicle_classes:
                    continue

                # Get vehicle class
                vehicle_class = vehicle_classes[class_name]

                # License plate recognition (crop the vehicle's plate)
                if class_name == 'license_plate':
                    plate_image = image_rgb[y1:y2, x1:x2]
                    plate_text = reader.readtext(plate_image, detail=0)
                    recognized_text = ''.join(plate_text).upper() if plate_text else "N/A"

                    # Draw bounding box on the image for plate
                    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box for plate number
                else:
                    recognized_text = "N/A"

    

                # Append to results storage
                st.session_state['results_data'].append(
                    {
                        "Datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
                        "Vehicle Class": vehicle_class,
                        "Plate Number": recognized_text,
                    }
                )

            # Display the image with YOLO detections (vehicles) 
            with col1:
 
                plate_image_rgb = image.copy()
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(plate_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box for plate number

                st.image(cv2.cvtColor(plate_image_rgb, cv2.COLOR_BGR2RGB), caption="Detected Plate Numbers", use_container_width=True)
          # Display the extracted plate image for OCR recognition
            with col2:          
                st.image(plate_image, caption="Extracted Plate Image for OCR", use_container_width=True)
            

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
