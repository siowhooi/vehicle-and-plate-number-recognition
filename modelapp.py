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

# Function to draw bounding boxes on an image
def draw_yolo_boxes(image, results, class_mapping):
    image_copy = image.copy()
    for result in results[0].boxes:
        class_id = int(result.cls)
        class_name = model.names[class_id]
        if class_name in class_mapping or class_name in ["license_plate", "license_plate_taxi"]:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            color = (0, 255, 0) if class_name in class_mapping else (0, 0, 255)  # Green for vehicles, red for plates
            label = class_mapping.get(class_name, class_name)
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image_copy

# Function to draw OCR bounding boxes
def draw_ocr_boxes(plate_image, ocr_results):
    plate_copy = plate_image.copy()
    for (bbox, text, prob) in ocr_results:
        (x_min, y_min), (x_max, y_max) = bbox[0], bbox[2]
        cv2.rectangle(plate_copy, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv2.putText(plate_copy, text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return plate_copy

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
    ocr_results = []

    # Mapping YOLO class names to human-readable class labels
    class_mapping = {
        "class0_emergencyVehicle": "Class 0",
        "class1_lightVehicle": "Class 1",
        "class2_mediumVehicle": "Class 2",
        "class3_heavyVehicle": "Class 3",
        "class4_taxi": "Class 4",
        "class5_bus": "Class 5"
    }

    # Draw bounding boxes for YOLO detections
    yolo_image = draw_yolo_boxes(image_rgb, results, class_mapping)

    # Iterate through the detected objects
    for result in results[0].boxes:
        class_id = int(result.cls)
        class_name = model.names[class_id]

        # Detect vehicle class
        if class_name in class_mapping:
            vehicle_class = class_mapping[class_name]

        # Detect license plate and crop it
        if class_name in ["license_plate", "license_plate_taxi"]:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            plate_image = image_rgb[y1:y2, x1:x2]

            # Perform OCR to recognize the text
            ocr_results = reader.readtext(plate_image)
            plate_text = ' '.join([text for (_, text, _) in ocr_results])

    # Draw OCR bounding boxes on the cropped plate image
    plate_with_boxes = None
    if plate_image is not None:
        plate_with_boxes = draw_ocr_boxes(plate_image, ocr_results)

    return vehicle_class, yolo_image, plate_with_boxes, plate_text

# Function to get toll fare based on the vehicle class and the selected entry/exit points
def get_toll_fare(vehicle_class, toll_plaza_type, entry_spot, exit_spot):
    # Toll rates for Open Toll System (Fixed)
    open_toll_fares = {
        "Class 1": 6.00,
        "Class 2": 12.00,
        "Class 3": 18.00,
        "Class 4": 3.00,
        "Class 5": 5.00,
        "Class 0": 0.00
    }

    # Toll rates for Closed Toll System (Distance-based)
    closed_toll_fares = {
        ("Jalan Duta, Kuala Lumpur", "Juru, Penang"): {
            "Class 1": 35.51,
            "Class 2": 64.90,
            "Class 3": 86.50,
            "Class 4": 17.71,
            "Class 5": 21.15
        },
        ("Jalan Duta, Kuala Lumpur", "Seremban, Negeri Sembilan"): {
            "Class 1": 10.58,
            "Class 2": 19.50,
            "Class 3": 29.50,
            "Class 4": 5.33,
            "Class 5": 7.95
        },
        ("Seremban, Negeri Sembilan", "Juru, Penang"): {
            "Class 1": 43.95,
            "Class 2": 80.50,
            "Class 3": 107.20,
            "Class 4": 22.06,
            "Class 5": 30.95
        }
    }

    if toll_plaza_type == "Open Toll System":
        return open_toll_fares.get(vehicle_class, 0.00)

    # For Closed Toll System, calculate fare based on entry and exit points
    if (entry_spot, exit_spot) in closed_toll_fares:
        return closed_toll_fares[(entry_spot, exit_spot)].get(vehicle_class, 0.00)
    return 0.00

# Streamlit app
st.title("Vehicle Class and License Plate Recognition")

# Sidebar for selecting toll plaza type
toll_plaza_type = st.sidebar.radio("Select Toll System", ["Open Toll System", "Closed Toll System"])

# Layout for two panels
col1, col2 = st.columns([2, 3])

# Initialize session state to store the entry spot and results list
if "entry_spot" not in st.session_state:
    st.session_state.entry_spot = None
    st.session_state.entry_class = None

if "results_data" not in st.session_state:
    st.session_state.results_data = []

# Left panel for image uploads or webcam captures
with col1:
    if toll_plaza_type == "Open Toll System":
        st.header("Open Toll System")
        spots = {1: None}  # Only one spot for Open Toll System
        spot_names = {1: "Gombak Toll Plaza"}
    else:
        st.header("Closed Toll System")
        # Dropdown to select entry location
        entry_location = st.selectbox("Select Entry Toll Plaza", ["Jalan Duta, Kuala Lumpur", "Seremban, Negeri Sembilan", "Juru, Penang"])

        st.subheader(f"Entry Location: {entry_location}")

    # Image upload or camera capture option
    option = st.radio(f"Select Detect Option:", ["Upload an Image", "Use Camera"])

    # File uploader or webcam input
    uploaded_file = None
    if option == "Upload an Image":
        uploaded_file = st.file_uploader(f"Upload image for {entry_location}", type=["jpg", "jpeg", "png"])
    elif option == "Use Camera":
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            uploaded_file = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Process the image when available
    if uploaded_file is not None:
        with st.spinner("Processing..."):
            vehicle_class, yolo_image, plate_with_boxes, plate_text = process_image(uploaded_file)

            # Get current time in Asia/Kuala_Lumpur timezone
            kuala_lumpur_tz = pytz.timezone('Asia/Kuala_Lumpur')
            current_time = datetime.now(kuala_lumpur_tz).strftime("%d/%m/%Y %H:%M")

            # Calculate toll fare based on entry/exit logic
            if toll_plaza_type == "Open Toll System":
                toll_fare = get_toll_fare(vehicle_class, toll_plaza_type, None, None)
            else:
                if st.session_state.entry_spot is None:
                    st.session_state.results_data.append({
                        "Datetime": current_time,
                        "Vehicle Class": vehicle_class,
                        "Plate Number": plate_text,
                        "Toll": entry_location,
                        "Mode": "Entry",
                        "Toll Fare (RM)": "-"
                    })
                    st.session_state.entry_spot = entry_location
                    st.session_state.entry_class = vehicle_class
                else:
                    exit_location = entry_location  # Assume exit is the same for now
                    toll_fare = get_toll_fare(vehicle_class, toll_plaza_type, st.session_state.entry_spot, exit_location)
                    st.session_state.results_data.append({
                        "Datetime": current_time,
                        "Vehicle Class": vehicle_class,
                        "Plate Number": plate_text,
                        "Toll": exit_location,
                        "Mode": "Exit",
                        "Toll Fare (RM)": toll_fare
                    })
                    st.session_state.entry_spot = None
                    st.session_state.entry_class = None

            # Show images with bounding boxes
            if yolo_image is not None:
                st.image(yolo_image, caption=f"Detected Vehicle - {entry_location}", use_column_width=True)

            if plate_with_boxes is not None:
                st.image(plate_with_boxes, caption=f"Detected Plate - {entry_location}", use_column_width=True)

# Right panel for displaying results
with col2:
    st.header("Detection Results")
    if st.session_state.results_data:
        st.table(st.session_state.results_data)
    else:
        st.info("No results available yet. Please upload or capture images.")
