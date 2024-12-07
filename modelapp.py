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

# Function to get toll fare based on the vehicle class
def get_toll_fare(vehicle_class, toll_plaza, spot_from, spot_to):
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

    if toll_plaza in open_toll_fares:
        return open_toll_fares.get(vehicle_class, 0.00)

    # For Closed Toll System, calculate fare based on entry and exit points
    if (spot_from, spot_to) in closed_toll_fares:
        return closed_toll_fares[(spot_from, spot_to)].get(vehicle_class, 0.00)
    return 0.00

# Streamlit app
st.title("Vehicle Class and License Plate Recognition")

# Toll plaza selection (no sidebar)
toll_plaza = st.selectbox("Select Toll Plaza", [
    "Gombak Toll Plaza",
    "Jalan Duta, Kuala Lumpur",
    "Seremban, Negeri Sembilan",
    "Juru, Penang"
])

# Layout for two panels
col1, col2 = st.columns([2, 3])

# Initialize session state to store the entry spot
if "entry_spot" not in st.session_state:
    st.session_state.entry_spot = None
    st.session_state.entry_class = None

# Left panel for image uploads or webcam captures
with col1:
    st.header(f"Selected Toll Plaza: {toll_plaza}")
    spots = {1: None}  # Only one spot for selection
    spot_names = {1: toll_plaza}

    results_data = []

    for spot_num in spots:
        spot_name = spot_names[spot_num]
        st.subheader(f"{spot_name}")
        option = st.radio(f"Select Detect Option:", ["Upload an Image", "Use Camera"], key=f"spot_{spot_num}")

        if option == "Upload an Image":
            uploaded_file = st.file_uploader(f"Upload image for {spot_name}", type=["jpg", "jpeg", "png"], key=f"file_{spot_num}")
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                spots[spot_num] = image

        elif option == "Use Camera":
            cap = cv2.VideoCapture(0)  
            ret, frame = cap.read()
            cap.release()
            if ret:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                spots[spot_num] = image
            else:
                st.warning(f"Failed to capture an image for {spot_name}")

        # Process the image dynamically as soon as it's available
        if spots[spot_num]:
            with st.spinner(f"Processing Spot {spot_name}..."):
                vehicle_class, yolo_image, plate_with_boxes, plate_text = process_image(spots[spot_num])
                if vehicle_class:
                    # Get current time in Asia/Kuala_Lumpur timezone
                    kuala_lumpur_tz = pytz.timezone('Asia/Kuala_Lumpur')
                    current_time = datetime.now(kuala_lumpur_tz).strftime("%d/%m/%Y %H:%M")

                    # Determine mode (Entry or Exit)
                    mode = "Entry" if st.session_state.entry_spot is None else "Exit"

                    # Calculate toll fare
                    toll_fare = "-"
                    if toll_plaza == "Gombak Toll Plaza":
                        toll_fare = get_toll_fare(vehicle_class, toll_plaza, "", "")
                    else:
                        if mode == "Exit":
                            toll_fare = get_toll_fare(vehicle_class, toll_plaza, st.session_state.entry_spot, spot_name)
                            st.session_state.entry_spot = None  # Reset entry after exit

                    # Save detection result
                    results_data.append({
                        "Datetime": current_time,
                        "Vehicle Class": vehicle_class,
                        "Plate Number": plate_text,
                        "Toll": spot_name,
                        "Mode": mode,
                        "Toll Fare (RM)": toll_fare
                    })
                    if yolo_image is not None:
                        st.image(yolo_image, caption=f"Detected Vehicle - {spot_name} with Bounding Boxes", use_column_width=True)

                    if plate_with_boxes is not None:
                        st.image(plate_with_boxes, caption=f"Detected Plate - {spot_name} with OCR Bounding Boxes", use_column_width=True)

# Right panel for displaying results
with col2:
    st.header("Detection Results")
    if results_data:
        st.table(results_data)
    else:
        st.info("No results available yet. Please upload or capture images.")
