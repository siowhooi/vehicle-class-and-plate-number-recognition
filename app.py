import streamlit as st
import cv2
import easyocr
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("path_to_your_yolov8_model/best.pt")

# Initialize EasyOCR reader for license plate recognition
reader = easyocr.Reader(['en'])

# Function to process the image
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

    # Show the image with YOLO detections
    st.image(image_rgb, channels='RGB', caption="Image with YOLO Detections", use_column_width=True)

    # Iterate through the detected objects
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

            # Display the cropped plate image with OCR bounding box
            st.image(plate_image, channels='RGB', caption="Cropped License Plate Image", use_column_width=True)

            # Print recognized plate number
            st.write(f"Recognized Plate Number: {''.join(plate_text)}")

# Streamlit App Layout
st.title("License Plate Recognition")
st.markdown("Upload an image to detect license plates and recognize plate numbers.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to a temporary image path
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the uploaded image
    process_image(image_path)
