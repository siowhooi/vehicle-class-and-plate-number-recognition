import streamlit as st
import cv2
import easyocr
import datetime
from ultralytics import YOLO
import matplotlib.pyplot as plt
from io import BytesIO

# Load the YOLO model
model = YOLO(r"C:\Users\wohen\PycharmProjects\test\runs\detect\train\weights\best.pt")

# Initialize EasyOCR reader for license plate recognition
reader = easyocr.Reader(['en'])

# Function to process and display results
def process_image(image_path):
    # Run inference on the image
    results = model(image_path)

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image with YOLO detections and bounding boxes
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
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    ax.axis('off')
    st.pyplot(fig)

    # OCR processing for license plates
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

            # Display the cropped plate image with OCR bounding box
            fig, ax = plt.subplots()
            ax.imshow(plate_image)
            ax.axis('off')
            st.pyplot(fig)

    # Return the recognized plate text
    return ''.join(plate_text)

# Function to display the toll fare table
def display_toll_fare(location, vehicle_class, plate_number):
    # Placeholder logic for toll fare based on location and vehicle type
    toll_fare = {
        "Gombak Toll Plaza": {"Car": 10, "Truck": 15},
        "Jalan Duta, Kuala Lumpur": {"Car": 12, "Truck": 18},
        "Seremban, Negeri Sembilan": {"Car": 8, "Truck": 14},
        "Juru, Penang": {"Car": 6, "Truck": 10},
    }

    # Get the current time and date
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get vehicle class (Placeholder logic, replace with actual vehicle type classification)
    vehicle_class = "Car" if vehicle_class is None else vehicle_class  # Assuming vehicle class is given or inferred

    # Calculate the toll fare based on location and vehicle type
    fare = toll_fare.get(location, {}).get(vehicle_class, "N/A")

    # Display the toll information
    st.write(f"### Toll Information")
    st.write(f"**Datetime:** {current_datetime}")
    st.write(f"**Toll Location:** {location}")
    st.write(f"**Vehicle Class:** {vehicle_class}")
    st.write(f"**Plate Number:** {plate_number}")
    st.write(f"**Mode:** {vehicle_class}")  # Can be replaced with mode info
    st.write(f"**Toll Fare (RM):** {fare}")

# Streamlit layout
st.title("Smart Tolling System")

# Upper Layout: 2x2 grid for locations
col1, col2 = st.columns(2)

# Location 1: Gombak Toll Plaza
with col1:
    st.header("Gombak Toll Plaza")
    uploaded_file_1 = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file_1 is not None:
        img_bytes = uploaded_file_1.read()
        image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        plate_number_1 = process_image(image)
        display_toll_fare("Gombak Toll Plaza", "Car", plate_number_1)

# Location 2: Jalan Duta, Kuala Lumpur
with col2:
    st.header("Jalan Duta, Kuala Lumpur")
    uploaded_file_2 = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file_2 is not None:
        img_bytes = uploaded_file_2.read()
        image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        plate_number_2 = process_image(image)
        display_toll_fare("Jalan Duta, Kuala Lumpur", "Car", plate_number_2)

# Lower Layout: 2x2 grid for other locations
col3, col4 = st.columns(2)

# Location 3: Seremban, Negeri Sembilan
with col3:
    st.header("Seremban, Negeri Sembilan")
    uploaded_file_3 = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file_3 is not None:
        img_bytes = uploaded_file_3.read()
        image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        plate_number_3 = process_image(image)
        display_toll_fare("Seremban, Negeri Sembilan", "Car", plate_number_3)

# Location 4: Juru, Penang
with col4:
    st.header("Juru, Penang")
    uploaded_file_4 = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file_4 is not None:
        img_bytes = uploaded_file_4.read()
        image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        plate_number_4 = process_image(image)
        display_toll_fare("Juru, Penang", "Car", plate_number_4)
