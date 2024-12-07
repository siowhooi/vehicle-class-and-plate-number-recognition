import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image

# Load YOLO model
model = YOLO("model/best.pt")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Streamlit app layout
st.set_page_config(layout="wide")  # Set layout to wide mode
st.title("Smart Tolling System")
st.write("Detect vehicles and recognize license plates for automated tolling.")

# Define the toll plaza options
toll_plazas = [
    "Gombak Toll Plaza",
    "Jalan Duta, Kuala Lumpur",
    "Seremban, Negeri Sembilan",
    "Juru, Penang",
]

# Create the layout
left_col, right_col = st.columns(2)

# Left side: Input options
with left_col:
    st.header("Input Options")

    # Dropdown menu for location selection
    selected_toll_plaza = st.selectbox("Select Toll Plaza", toll_plazas)

    # File uploader or camera input
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    use_camera = st.button("Open Camera (Coming Soon)")

# Right side: Display results
with right_col:
    st.header("Detection Results")

    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Run YOLO inference
        results = model(image_np)

        # Convert to RGB for OpenCV processing
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Draw detections
        for result in results[0].boxes:
            class_id = int(result.cls)
            class_name = model.names[class_id]

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, result.xyxy[0])

            # Draw bounding boxes
            color = (0, 255, 0) if class_name != 'license_plate' else (255, 0, 0)
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_rgb, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Perform OCR on license plates
            if class_name == 'license_plate':
                plate_image = image_rgb[y1:y2, x1:x2]
                plate_text = reader.readtext(plate_image, detail=0)
                st.write(f"Detected Plate: {''.join(plate_text)}")

        # Display the image with detections
        st.image(image_rgb, caption='Detected Image', use_column_width=True)

    elif use_camera:
        st.write("Camera feature is under development.")

    else:
        st.write("Upload an image to begin detection.")

# Footer
st.write("---")
st.write(f"You selected: **{selected_toll_plaza}**")
