import streamlit as st
from ultralytics import YOLO
import easyocr
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
from datetime import datetime

# Load YOLO model
model = YOLO(r"C:\Users\wohen\PycharmProjects\test\runs\detect\train\weights\best.pt")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Toll Plaza locations
locations = [
    "Gombak Toll Plaza",
    "Jalan Duta, Kuala Lumpur",
    "Seremban, Negeri Sembilan",
    "Juru, Penang"
]

# Function to recognize plate and vehicle class
def detect_vehicle_and_plate(image):
    # Run inference on the image
    results = model(image)

    # Prepare the image for display
    image_rgb = np.array(image.convert('RGB'))

    plate_number = ""
    vehicle_class = ""

    # Process the detected objects
    for result in results[0].boxes:
        class_id = int(result.cls)
        class_name = model.names[class_id]

        # Get the bounding box coordinates
        x1, y1, x2, y2 = map(int, result.xyxy[0])

        # Detect vehicle class and license plate
        if class_name == "license_plate" or class_name == "license_plate_taxi":
            # Crop the license plate from the image
            plate_image = image_rgb[y1:y2, x1:x2]
            plate_text = reader.readtext(plate_image, detail=0)
            plate_number = ''.join(plate_text)
            
            # Draw bounding box for license plate on the image
            plt.imshow(plate_image)
            plt.axis('off')
            st.pyplot()

        vehicle_class = class_name

    return vehicle_class, plate_number, image_rgb

# Layout for the Streamlit app
st.title("Tolling System with Vehicle and Plate Recognition")

# Left column (for location selection, image upload, or camera)
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Choose Toll Plaza Location", locations)

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        vehicle_class, plate_number, result_image = detect_vehicle_and_plate(image)

        st.image(result_image, caption="Processed Image", use_column_width=True)

        st.write(f"Location: {location}")
        st.write(f"Datetime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"Vehicle Class: {vehicle_class}")
        st.write(f"Plate Number: {plate_number}")
        st.write(f"Toll: {location}")
        st.write("Mode: Upload Image")
        st.write(f"Toll Fare (RM): {calculate_toll_fare(location)}")

with col2:
    st.subheader("Result Table")
    if uploaded_file:
        st.write(
            f"""
            | Datetime            | Vehicle Class | Plate Number | Toll         | Mode       | Toll Fare (RM) |
            |---------------------|---------------|--------------|--------------|------------|----------------|
            | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {vehicle_class} | {plate_number}  | {location}   | Upload Image | {calculate_toll_fare(location)} |
            """
        )

# Function to calculate toll fare based on location
def calculate_toll_fare(location):
    toll_fares = {
        "Gombak Toll Plaza": 5.00,
        "Jalan Duta, Kuala Lumpur": 7.00,
        "Seremban, Negeri Sembilan": 4.50,
        "Juru, Penang": 6.00
    }
    return toll_fares.get(location, 0.00)

