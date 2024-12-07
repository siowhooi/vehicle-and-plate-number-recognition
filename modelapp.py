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
        class_name = model.names[class_id]

        if class_name in ['license_plate', 'license_plate_taxi']:
            # Extract the bounding box coordinates
            x1, y1, x2, y2 = map(int, result.xyxy[0])

            # Crop the license plate from the image
            plate_image = image_rgb[y1:y2, x1:x2]

            # Perform OCR to recognize the text
            plate_text = reader.readtext(plate_image, detail=0)

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
            class_mapping = {
                "class0_emergencyVehicle": "Class 0",
                "class1_lightVehicle": "Class 1",
                "class2_mediumVehicle": "Class 2",
                "class3_heavyVehicle": "Class 3",
                "class4_taxi": "Class 4",
                "class5_bus": "Class 5"
            }
            vehicle_class_label = class_mapping.get(vehicle_class, "Unknown")

            # Append result to the DataFrame
            result_data.append({
                "Datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "Vehicle Class": vehicle_class_label,
                "Plate Number": plate_text,
                "Toll": location,
                "Mode": "Entry",  # Assuming "Entry" mode for now, can be adjusted as needed
                "Toll Fare (RM)": "-"  # Placeholder for toll fare
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
                class_mapping = {
                    "class0_emergencyVehicle": "Class 0",
                    "class1_lightVehicle": "Class 1",
                    "class2_mediumVehicle": "Class 2",
                    "class3_heavyVehicle": "Class 3",
                    "class4_taxi": "Class 4",
                    "class5_bus": "Class 5"
                }
                vehicle_class_label = class_mapping.get(vehicle_class, "Unknown")

                # Append result to the DataFrame
                result_data.append({
                    "Datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "Vehicle Class": vehicle_class_label,
                    "Plate Number": plate_text,
                    "Toll": location,
                    "Mode": "Entry",  # Assuming "Entry" mode for now, can be adjusted as needed
                    "Toll Fare (RM)": "-"  # Placeholder for toll fare
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
