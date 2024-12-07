import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import matplotlib.pyplot as plt
from datetime import datetime

# Load the YOLO model
model = YOLO(r"best.pt")

# Initialize EasyOCR reader for license plate recognition
reader = easyocr.Reader(['en'])

# Dropdown menu for selecting location
locations = ['Gombak Toll Plaza', 'Jalan Duta, Kuala Lumpur', 'Seremban, Negeri Sembilan', 'Juru, Penang']
location = st.selectbox("Select Toll Location", locations)

# Upload or capture image
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if st.button("Capture Image via Camera"):
    # Code for opening webcam to capture image goes here (if applicable)

    # Placeholder for webcam capture (as webcam implementation may need extra configuration)
    st.write("Webcam capture functionality can be implemented here.")

if image_file is not None:
    # Convert uploaded image to OpenCV format
    image_bytes = image_file.read()
    image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(image_array, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run inference on the image
    results = model(image)

    # Display image with YOLO detections and bounding boxes for each detected object
    for result in results[0].boxes:
        class_id = int(result.cls)
        class_name = model.names[class_id]

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, result.xyxy[0])

        # Draw the bounding box on the image for detected objects
        color = (0, 255, 0) if class_name != 'license_plate' else (255, 0, 0)
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_rgb, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Show the image with detections
    st.image(image_rgb, caption="Detected Image", use_column_width=True)

    # Process detected license plates and perform OCR
    plate_text = ""
    for result in results[0].boxes:
        class_id = int(result.cls)
        class_name = model.names[class_id]

        if class_name in ['license_plate', 'license_plate_taxi']:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            plate_image = image_rgb[y1:y2, x1:x2]

            # Use EasyOCR to recognize text from the plate
            plate_text = reader.readtext(plate_image, detail=0)

            # Draw bounding box around OCR text
            for (bbox, text, _) in reader.readtext(plate_image):
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))
                cv2.rectangle(plate_image, top_left, bottom_right, (0, 255, 0), 2)

            # Display cropped plate image
            st.image(plate_image, caption="License Plate Detected", use_column_width=True)

            # Show recognized plate number
            st.write(f"Recognized Plate Number: {''.join(plate_text)}")

    # Display result table on the right side
    if plate_text:
        # Sample toll fares (can be replaced by actual logic based on the location)
        toll_fare = 10.0 if location == 'Gombak Toll Plaza' else 15.0  # Example value
        data = {
            "Datetime": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            "Vehicle Class": [class_name],
            "Plate Number": [''.join(plate_text)],
            "Toll": [location],
            "Mode": ['Manual' if image_file else 'Camera'],
            "Toll Fare (RM)": [toll_fare]
        }

        st.table(data)
