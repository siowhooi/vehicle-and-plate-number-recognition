import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

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
            vehicle_class = class_name

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
toll_plaza_type = st.sidebar.radio("Select Toll System Types", ["Open Toll System", "Closed Toll System"])

# Layout for two panels
col1, col2 = st.columns([2, 3])

# Left panel for image uploads or webcam captures
with col1:
    if toll_plaza_type == "Open Toll Plaza":
        st.header("Open Toll Plaza and Detected Vehicle")
    elif toll_plaza_type == "Closed Toll Plaza":
        st.header("Closed Toll Plaza and Detected Vehicle")

    spots = {}
    results_data = []

    for i in range(1, 5):
        st.subheader(f"Toll Plaza Spot {i}")
        option = st.radio(f"Detected Vehicle {i}:", ["Upload an Image", "Use Webcam"], key=f"spot_{i}")

        if option == "Upload an Image":
            uploaded_file = st.file_uploader(f"Upload image for Spot {i}", type=["jpg", "jpeg", "png"], key=f"file_{i}")
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded Image - Spot {i}", use_column_width=True)
                spots[i] = image

        elif option == "Use Webcam":
            # Automatically capture the image
            cap = cv2.VideoCapture(0)  # Use webcam (0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                st.image(image, caption=f"Captured Image - Spot {i}", use_column_width=True)
                spots[i] = image
            else:
                st.warning(f"Failed to capture an image for Spot {i}")

        # Process the image dynamically as soon as it's available
        if i in spots:
            with st.spinner(f"Processing Spot {i}..."):
                vehicle_class, plate_image, plate_text = process_image(spots[i])
                if vehicle_class:
                    # Save detection result
                    results_data.append({
                        "datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
                        "vehicle_class": vehicle_class,
                        "plate_number": plate_text,
                        "spot": i
                    })
                    if plate_image is not None:
                        st.image(plate_image, caption=f"Detected Plate - Spot {i}", use_column_width=True)

# Right panel for displaying results
with col2:
    st.header("Detection Results")
    if results_data:
        st.table(results_data)
    else:
        st.info("No results available yet. Please upload or capture images.")
