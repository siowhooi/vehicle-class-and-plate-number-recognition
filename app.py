import streamlit as st
import cv2
import easyocr
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

# Streamlit Title
st.title("Vehicle and License Plate Recognition")

# Upload image through Streamlit file uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load the YOLO model
    model = YOLO(r"best.pt")  # Update path for your GitHub or local directory

    # Read the uploaded image
    image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference on the image
    results = model(image)

    # Initialize EasyOCR reader for license plate recognition
    reader = easyocr.Reader(['en'])

    # Draw YOLO bounding boxes on image
    for result in results[0].boxes:
        class_id = int(result.cls)
        class_name = model.names[class_id]
        x1, y1, x2, y2 = map(int, result.xyxy[0])

        color = (0, 255, 0) if class_name != 'license_plate' else (255, 0, 0)
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_rgb, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # For license plates, crop and use EasyOCR
        if class_name in ['license_plate', 'license_plate_taxi']:
            plate_image = image_rgb[y1:y2, x1:x2]
            plate_text = reader.readtext(plate_image, detail=0)

            # Draw bounding boxes for OCR detection
            for (bbox, text, _) in reader.readtext(plate_image):
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))
                cv2.rectangle(plate_image, top_left, bottom_right, (0, 255, 0), 2)

            # Show the cropped plate image with OCR
            st.image(plate_image, caption="Cropped License Plate", use_column_width=True)
            st.write(f"Recognized Plate Number: {''.join(plate_text)}")

    # Display the image with YOLO detections
    st.image(image_rgb, caption="Detected Vehicles and License Plates", use_column_width=True)
