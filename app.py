import streamlit as st
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import easyocr
from datetime import datetime

# Initialize YOLO model
model = YOLO(r"best.pt")
# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Helper function to run YOLO and EasyOCR
def process_image(image_path):
    results = model(image_path)

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image with YOLO detections and bounding boxes for each detected object
    for result in results[0].boxes:
        class_id = int(result.cls)
        class_name = model.names[class_id]
        x1, y1, x2, y2 = map(int, result.xyxy[0])

        # Draw the bounding box on the image for the detected object
        color = (0, 255, 0) if class_name != 'license_plate' else (255, 0, 0)  # Green for vehicles, Red for license plates
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_rgb, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Initialize variable for cropped plate text
    plate_text = ""

    # Iterate through the detected objects
    for result in results[0].boxes:
        class_id = int(result.cls)
        class_name = model.names[class_id]

        # Check if the detected object is a license plate
        if class_name in ['license_plate', 'license_plate_taxi']:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            plate_image = image_rgb[y1:y2, x1:x2]

            # Use EasyOCR to recognize text from the plate image
            plate_text = reader.readtext(plate_image, detail=0)

            # Draw the bounding box on the cropped plate image for OCR detection
            for (bbox, text, _) in reader.readtext(plate_image):
                (top_left, top_right, bottom_right, bottom_left) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))
                cv2.rectangle(plate_image, top_left, bottom_right, (0, 255, 0), 2)

            return image_rgb, plate_image, ''.join(plate_text)

    return image_rgb, None, plate_text

# Streamlit layout
st.title('Smart Tolling System with Computer Vision')

# Upper Layout: Locations and Image Upload
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Define locations
locations = ['Gombak Toll Plaza', 'Jalan Duta, Kuala Lumpur', 'Seremban, Negeri Sembilan', 'Juru, Penang']

# Upload image and process for each location
image_paths = {}
for loc, col in zip(locations, [col1, col2, col3, col4]):
    with col:
        st.subheader(loc)
        uploaded_file = st.file_uploader(f"Upload image for {loc}", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_path = f"temp_{loc}.jpg"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            image_with_detections, plate_image, plate_text = process_image(image_path)

            # Display images
            st.image(image_with_detections, caption=f"YOLO Detection ({loc})", use_column_width=True)
            if plate_image is not None:
                st.image(plate_image, caption=f"Cropped Plate ({loc})", use_column_width=True)
            st.write(f"Recognized Plate Number: {plate_text}")

# Lower Layout: Toll Details
st.subheader('Toll Information')
toll_fare = 10  # Placeholder, update with logic based on location or other factors
mode = "Manual"  # Placeholder for toll mode (can be based on detection or user input)

# Get current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Display the table
st.table({
    'Datetime': [current_datetime],
    'Vehicle Class': ['Car'],  # Placeholder, update based on YOLO detection
    'Plate Number': ['ABC1234'],  # Placeholder, update based on OCR
    'Toll': locations[0],  # Update based on detected location
    'Mode': [mode],
    'Toll Fare (RM)': [toll_fare]
})
