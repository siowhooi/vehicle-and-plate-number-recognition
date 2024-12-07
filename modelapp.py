import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime

# Load YOLO model
model = YOLO(r"best.pt")  

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Toll fare rates
gombak_toll_rates = {
    "Class 1": 6.00,
    "Class 2": 12.00,
    "Class 3": 18.00,
    "Class 4": 3.00,
    "Class 5": 5.00
}

# Other toll fare rates
other_toll_rates = {
    ("Jalan Duta, Kuala Lumpur", "Juru, Penang"): {
        "Class 1": 35.51,
        "Class 2": 64.90,
        "Class 3": 86.50,
        "Class 4": 17.71,
        "Class 5": 21.15
    },
    ("Seremban, Negeri Sembilan", "Jalan Duta, Kuala Lumpur"): {
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

# Initialize a dictionary to track entry and exit events
entry_exit_tracker = {}

# Class mapping for vehicle classes
class_mapping = {
    "class0_emergencyVehicle": "Class 0",
    "class1_lightVehicle": "Class 1",
    "class2_mediumVehicle": "Class 2",
    "class3_heavyVehicle": "Class 3",
    "class4_taxi": "Class 4",
    "class5_bus": "Class 5"
}

# Function to calculate toll fare based on the vehicle class and location
def calculate_toll_fare(vehicle_class_label, location):
    if location == "Gombak Toll Plaza":
        # Fixed rates for Gombak Toll Plaza
        return gombak_toll_rates.get(vehicle_class_label, 0.00)
    else:
        # Dynamic rates for other toll locations
        entry_exit_key = (location, entry_exit_tracker.get(vehicle_class_label, {}).get("last_location"))
        
        # If it's an entry point, store the entry location
        if entry_exit_key not in other_toll_rates:
            entry_exit_tracker[vehicle_class_label] = {
                "last_location": location,
                "mode": "Entry"
            }
            return "-"  # Still waiting for exit to calculate fare
        else:
            # It's an exit, calculate toll fare
            toll_fare = other_toll_rates[entry_exit_key].get(vehicle_class_label, 0.00)
            entry_exit_tracker[vehicle_class_label] = {"mode": "Exit"}
            return toll_fare

# Function to process image and recognize license plate
def process_image(image):
    # Convert PIL Image to OpenCV format
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Run YOLO inference
    results = model(image_rgb)

    # Iterate through the detected objects
    for result in results[0].boxes:
        class_id = int(result.cls)
        class_name = model.names[class_id]  # Get the class name from the model
        
        # Check if the class name is in the mapping
        if class_name in class_mapping:
            # Extract the bounding box coordinates
            x1, y1, x2, y2 = map(int, result.xyxy[0])

            # Crop the license plate from the image
            plate_image = image_rgb[y1:y2, x1:x2]

            # Perform OCR to recognize the text
            plate_text = reader.readtext(plate_image, detail=0)

            # Return detected plate and class
            return plate_image, ''.join(plate_text), class_name

    return None, "No license plate detected", None

# Create a DataFrame to store results
result_data = []

# Streamlit app
st.title("License Plate Recognition")

# Left interface (location and image selection)
st.subheader("Location and Image Capture")

# Location selection
location = st.selectbox("Select Toll Location", [
    "Gombak Toll Plaza", 
    "Jalan Duta, Kuala Lumpur", 
    "Seremban, Negeri Sembilan", 
    "Juru, Penang"
])

# Image upload or webcam capture
option = st.radio("Choose Input Method:", ("Upload an Image", "Use Webcam"))

if option == "Upload an Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image
        with st.spinner("Processing..."):
            plate_image, plate_text, vehicle_class = process_image(image)

        # Display results
        if plate_image is not None:
            st.image(plate_image, caption="Detected License Plate", use_column_width=True)
            # Map vehicle class
            vehicle_class_label = class_mapping.get(vehicle_class, "Unknown")

            # Handle toll fare calculation
            toll_fare = calculate_toll_fare(vehicle_class_label, location)
            
            # Append result to the DataFrame
            result_data.append({
                "Datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "Vehicle Class": vehicle_class_label,
                "Plate Number": plate_text,
                "Toll": location,
                "Mode": "Entry",  # Assuming "Entry" mode for now, can be adjusted as needed
                "Toll Fare (RM)": toll_fare
            })

        else:
            st.warning("No license plate detected in the image.")
elif option == "Use Webcam":
    # Webcam capture
    run = st.button("Start Webcam")
    if run:
        # Open webcam and capture a frame
        cap = cv2.VideoCapture(0)  # Use webcam (0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Convert to PIL format
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.image(image, caption="Captured Image", use_column_width=True)

            # Process the image
            with st.spinner("Processing..."):
                plate_image, plate_text, vehicle_class = process_image(image)

            # Display results
            if plate_image is not None:
                st.image(plate_image, caption="Detected License Plate", use_column_width=True)
                # Map vehicle class
                vehicle_class_label = class_mapping.get(vehicle_class, "Unknown")

                # Handle toll fare calculation
                toll_fare = calculate_toll_fare(vehicle_class_label, location)
                
                # Append result to the DataFrame
                result_data.append({
                    "Datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "Vehicle Class": vehicle_class_label,
                    "Plate Number": plate_text,
                    "Toll": location,
                    "Mode": "Entry",  # Assuming "Entry" mode for now, can be adjusted as needed
                    "Toll Fare (RM)": toll_fare
                })
            else:
                st.warning("No license plate detected in the image.")
        else:
            st.error("Failed to capture an image. Please try again.")

# Right interface (result display)
st.subheader("Recognition Results")

# Display results in a table
if result_data:
    df = pd.DataFrame(result_data)
    st.dataframe(df)
else:
    st.write("No results to display.")
