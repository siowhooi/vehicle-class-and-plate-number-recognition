import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

# Function to read and display image
def display_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    st.pyplot(fig)

# Load the YOLO model
model = YOLO(r"best.pt")  # Replace with your actual model path

# Streamlit UI
st.title('Vehicle License Plate Detection and Recognition')
st.markdown('Upload an image of a vehicle, and the app will detect the license plate and recognize its text.')

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image_bytes = uploaded_file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Run inference using YOLO
    results = model(image)

    # Load EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Process the image with YOLO detections
    for result in results[0].boxes:
        class_id = int(result.cls)
        class_name = model.names[class_id]

        # Get the bounding box coordinates
        x1, y1, x2, y2 = map(int, result.xyxy[0])

        # Draw bounding box and label
        color = (0, 255, 0) if class_name != 'license_plate' else (255, 0, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Recognize plate text if it's a license plate
        if class_name in ['license_plate', 'license_plate_taxi']:
            plate_image = image[y1:y2, x1:x2]
            plate_text = reader.readtext(plate_image, detail=0)
            st.write(f"Recognized Plate Number: {''.join(plate_text)}")

    # Display the image with bounding boxes
    display_image(image)
