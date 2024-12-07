import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tempfile

# Load the YOLO model
model = YOLO(r"C:\Users\wohen\PycharmProjects\test\runs\detect\train\weights\best.pt")

# Initialize EasyOCR reader for license plate recognition
reader = easyocr.Reader(['en'])

def load_image(image_file):
    # Load image as a PIL object
    image = Image.open(image_file)
    return np.array(image)

def process_image(image_path):
    # Run inference on the image
    results = model(image_path)

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image with YOLO detections and bounding boxes for each detected object
    for result in results[0].boxes:
        class_id = int(result.cls)
        class_name = model.names[class_id]

        # Get the bounding box coordinates
        x1, y1, x2, y2 = map(int, result.xyxy[0])

        # Draw the bounding box on the image for the detected object
        color = (0, 255, 0) if class_name != 'license_plate' else (255, 0, 0)  # Green for vehicles, Red for license plates
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)  # Draw the rectangle on the image
        cv2.putText(image_rgb, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # Label the bounding box

    return image_rgb, results

def extract_plate_number(results, image_rgb):
    plate_text = ""
    for result in results[0].boxes:
        class_id = int(result.cls)
        class_name = model.names[class_id]

        # Check if the detected object is a license plate
        if class_name in ['license_plate', 'license_plate_taxi']:
            # Extract the coordinates of the bounding box
            x1, y1, x2, y2 = map(int, result.xyxy[0])

            # Crop the license plate from the image
            plate_image = image_rgb[y1:y2, x1:x2]

            # Use EasyOCR to recognize text from the plate image
            plate_text = reader.readtext(plate_image, detail=0)  # Recognize text without bounding box details

            # Draw the bounding box on the cropped plate image for OCR detection
            for (bbox, text, _) in reader.readtext(plate_image):
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))
                cv2.rectangle(plate_image, top_left, bottom_right, (0, 255, 0), 2)  # Draw the bounding box for OCR text

            return plate_text, plate_image
    return plate_text, None

# Streamlit UI
st.title('Vehicle and License Plate Recognition')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_image_path = temp_file.name
    
    # Process the image
    image_rgb, results = process_image(temp_image_path)
    
    # Display the image with YOLO bounding boxes
    st.image(image_rgb, caption='Processed Image', use_column_width=True)

    # Extract and display the plate number
    plate_text, plate_image = extract_plate_number(results, image_rgb)
    if plate_text:
        st.write(f"Recognized Plate Number: {''.join(plate_text)}")
        
        # Show the cropped plate image with bounding boxes for OCR
        st.image(plate_image, caption='Cropped Plate Image with OCR', use_column_width=True)
    else:
        st.write("No license plate detected.")
