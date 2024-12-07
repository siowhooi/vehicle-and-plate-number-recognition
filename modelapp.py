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

# Mapping YOLO classes to display names and toll fares
class_mapping = {
    "class0_emergencyVehicle": ("Class 0", 0.00),
    "class1_lightVehicle": ("Class 1", 6.00),
    "class2_mediumVehicle": ("Class 2", 12.00),
    "class3_heavyVehicle": ("Class 3", 18.00),
    "class4_taxi": ("Class 4", 3.00),
    "class5_bus": ("Class 5", 5.00)
}

# Closed Toll System fares by entry and exit locations
closed_toll_rates = {
    ("Jalan Duta", "Juru"): {"Class 1": 35.51, "Class 2": 64.90, "Class 3": 86.50, "Class 4": 17.71, "Class 5": 21.15},
    ("Seremban", "Jalan Duta"): {"Class 1": 10.58, "Class 2": 19.50, "Class 3": 29.50, "Class 4": 5.33, "Class 5": 7.95},
    ("Seremban", "Juru"): {"Class 1": 43.95, "Class 2": 80.50, "Class 3": 107.20, "Class 4": 22.06, "Class 5": 30.95}
}

# Reverse the rates for both directions
for (entry, exit), rates in list(closed_toll_rates.items()):
    closed_toll_rates[(exit, entry)] = rates

# Track entry points for closed toll system
entry_points = {}

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
        if class_name in class_mapping:
            vehicle_class = class_mapping[class_name][0]

        # Detect license plate and crop it
        if class_name in ["license_plate", "license_plate_taxi"]:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            plate_image = image_rgb[y1:y2, x1:x2]

            # Perform OCR to recognize the text
            plate_text = ''.join(reader.readtext(plate_image, detail=0))

    return vehicle_class, plate_image, plate_text

# Function to calculate toll fare
def calculate_toll_fare(toll_type, spot_name, vehicle_class, plate_number):
    if vehicle_class == "Class 0":
        return 0.00  # Free for Class 0 vehicles

    if toll_type == "Open Toll System":
        return class_mapping[f"class{vehicle_class[-1]}"][1]  # Use fixed fare mapping

    elif toll_type == "Closed Toll System":
        if plate_number not in entry_points:
            # First detection: record as entry
            entry_points[plate_number] = spot_name
            return None  # No toll fare until exit

        else:
            # Second detection: record as exit and calculate fare
            entry_point = entry_points.pop(plate_number)  # Get and remove entry point
            if (entry_point, spot_name) in closed_toll_rates:
                return closed_toll_rates[(entry_point, spot_name)].get(vehicle_class, 0.00)
            return 0.00  # Default toll fare

# Streamlit app
st.title("Vehicle Class and License Plate Recognition")

# Sidebar for selecting toll plaza type
toll_plaza_type = st.sidebar.radio("Select Toll System Types", ["Open Toll System", "Closed Toll System"])

# Layout for two panels
col1, col2 = st.columns([2, 3])

# Left panel for image uploads or webcam captures
with col1:
    if toll_plaza_type == "Open Toll System":
        st.header("Open Toll System")
        spots = {1: "Gombak Toll Plaza"}  # Only one spot for Open Toll System
    else:
        st.header("Closed Toll System")
        spots = {1: "Jalan Duta", 2: "Juru", 3: "Seremban"}  # Three spots for Closed Toll System

    results_data = []

    for spot_num, spot_name in spots.items():
        st.subheader(f"{spot_name}")
        option = st.radio(f"Detected Vehicle at {spot_name}:", ["Upload an Image", "Use Webcam"], key=f"spot_{spot_num}")

        if option == "Upload an Image":
            uploaded_file = st.file_uploader(f"Upload image for {spot_name}", type=["jpg", "jpeg", "png"], key=f"file_{spot_num}")
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded Image - {spot_name}", use_column_width=True)

        elif option == "Use Webcam":
            # Automatically capture the image
            cap = cv2.VideoCapture(0)  # Use webcam (0)
            ret, frame = cap.read()
            cap.release()
            if ret:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                st.image(image, caption=f"Captured Image - {spot_name}", use_column_width=True)

        # Process the image dynamically as soon as it's available
        if locals().get("image"):
            with st.spinner(f"Processing Spot {spot_name}..."):
                vehicle_class, plate_image, plate_text = process_image(image)
                if vehicle_class:
                    # Get current time in Asia/Kuala_Lumpur timezone
                    kuala_lumpur_tz = pytz.timezone('Asia/Kuala_Lumpur')
                    current_time = datetime.now(kuala_lumpur_tz).strftime("%d/%m/%Y %H:%M")

                    # Calculate toll fare
                    toll_fare = calculate_toll_fare(toll_plaza_type, spot_name, vehicle_class, plate_text)

                    # Save detection result
                    results_data.append({
                        "datetime": current_time,
                        "spot": spot_name,
                        "vehicle_class": vehicle_class,
                        "plate_number": plate_text,
                        "toll_fare": f"RM{toll_fare:.2f}" if toll_fare is not None else "Entry Recorded"
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
