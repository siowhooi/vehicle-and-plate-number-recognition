import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import pytz

# Load YOLO model
model = YOLO(r"best.pt")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to process image and detect vehicle class and license plate
def process_image(image):
    # Convert PIL Image to OpenCV format
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Run YOLO inference
    results = model(image_rgb)

    vehicle_class = None
    plate_image = None
    plate_text = "No license plate detected"

    # Iterate through the detected objects
    for result in results[0].boxes:
        class_id = int(result.cls)
        class_name = model.names[class_id]

        # Detect vehicle class
        if class_name in [
            "class0_emergencyVehicle", "class1_lightVehicle",
            "class2_mediumVehicle", "class3_heavyVehicle",
            "class4_taxi", "class5_bus"
        ]:
            vehicle_class = {
                "class0_emergencyVehicle": "Class 0",
                "class1_lightVehicle": "Class 1",
                "class2_mediumVehicle": "Class 2",
                "class3_heavyVehicle": "Class 3",
                "class4_taxi": "Class 4",
                "class5_bus": "Class 5"
            }.get(class_name)

        # Detect license plate and crop it
        if class_name in ["license_plate", "license_plate_taxi"]:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            plate_image = image_rgb[y1:y2, x1:x2]

            # Perform OCR to recognize the text
            plate_text = ''.join(reader.readtext(plate_image, detail=0))

    return vehicle_class, plate_image, plate_text

# Streamlit app
st.title("Vehicle Class and License Plate Recognition")

# Sidebar for selecting toll plaza type
toll_plaza_type = st.sidebar.radio("Select Toll System", ["Open Toll System", "Closed Toll System"])

# Layout for two panels
col1, col2 = st.columns([2, 3])

# Left panel for image uploads or webcam captures
with col1:
    if toll_plaza_type == "Open Toll System":
        st.header("Open Toll System")
        spots = {1: None}  # Only one spot for Open Toll System
        spot_names = {1: "Gombak Toll Plaza"}
    else:
        st.header("Closed Toll System")
        spots = {1: None, 2: None, 3: None}  # Three spots for Closed Toll System
        spot_names = {
            1: "Jalan Duta, Kuala Lumpur",
            2: "Seremban, Negeri Sembilan",
            3: "Juru, Penang"
        }

    results_data = []

    for spot_num in spots:
        spot_name = spot_names[spot_num]
        st.subheader(f"{spot_name}")
        option = st.radio(f"Select Detect Option:", ["Upload an Image", "Use Camera"], key=f"spot_{spot_num}")

        if option == "Upload an Image":
            uploaded_file = st.file_uploader(f"Upload image for {spot_name}", type=["jpg", "jpeg", "png"], key=f"file_{spot_num}")
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Detected Vehicle - {spot_name}", use_column_width=True)
                spots[spot_num] = image

        elif option == "Use Camera":
            # Automatically capture the image
            cap = cv2.VideoCapture(0)  
            ret, frame = cap.read()
            cap.release()
            if ret:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                st.image(image, caption=f"Captured Image - {spot_name}", use_column_width=True)
                spots[spot_num] = image
            else:
                st.warning(f"Failed to capture an image for {spot_name}")

        # Process the image dynamically as soon as it's available
        if spots[spot_num]:
            with st.spinner(f"Processing Spot {spot_name}..."):
                vehicle_class, plate_image, plate_text = process_image(spots[spot_num])
                if vehicle_class:
                    # Get current time in Asia/Kuala_Lumpur timezone
                    kuala_lumpur_tz = pytz.timezone('Asia/Kuala_Lumpur')
                    current_time = datetime.now(kuala_lumpur_tz).strftime("%d/%m/%Y %H:%M")

                    # Save detection result
                    results_data.append({
                        "Datetime": current_time,
                        "Vehicle Class": vehicle_class,
                        "Plate Number": plate_text,
                        "Toll": spot_name
                    })
                    if plate_image is not None:
                        st.image(plate_image, caption=f"Detected Plate - {spot_name}", use_column_width=True)

# Right panel for displaying results
with col2:
    st.header("Detection Results")
    if results_data:
        st.table(results_data)
    else:
        st.info("No results available yet. Please upload or capture images.")
