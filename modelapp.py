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

            return plate_image, class_name, ''.join(plate_text)

    return None, None, "No license plate detected"

# Streamlit app
st.title("Multi-Spot License Plate Recognition")

# Dynamic layout for flexibility
spots_container = st.container()
results_container = st.container()

with spots_container:
    st.header("Input Spots")
    spots = {}
    spot_columns = st.columns(2)  # Two spots per row

    for i in range(1, 5):
        with spot_columns[(i - 1) % 2]:  # Arrange spots in 2 columns dynamically
            st.subheader(f"Spot {i}")
            option = st.radio(f"Input for Spot {i}:", ["Upload an Image", "Use Webcam"], key=f"spot_{i}")

            if option == "Upload an Image":
                uploaded_file = st.file_uploader(f"Upload image for Spot {i}", type=["jpg", "jpeg", "png"], key=f"file_{i}")
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Uploaded Image - Spot {i}", use_column_width=True)
                    spots[i] = image

            elif option == "Use Webcam":
                capture = st.button(f"Capture Image for Spot {i}", key=f"capture_{i}")
                if capture:
                    cap = cv2.VideoCapture(0)  # Use webcam (0)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        st.image(image, caption=f"Captured Image - Spot {i}", use_column_width=True)
                        spots[i] = image
                    else:
                        st.warning(f"Failed to capture an image for Spot {i}")

with results_container:
    st.header("Detection Results")
    if st.button("Process All Spots"):
        results_data = []
        for spot, image in spots.items():
            if image:
                with st.spinner(f"Processing Spot {spot}..."):
                    plate_image, vehicle_class, plate_text = process_image(image)
                    if plate_image is not None:
                        # Save detection result
                        results_data.append({
                            "datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
                            "vehicle_class": vehicle_class,
                            "plate_number": plate_text,
                            "spot": spot
                        })
                        st.image(plate_image, caption=f"Detected Plate - Spot {spot}", use_column_width=True)
                    else:
                        results_data.append({
                            "datetime": datetime.now().strftime("%d/%m/%Y %H:%M"),
                            "vehicle_class": "N/A",
                            "plate_number": "No license plate detected",
                            "spot": spot
                        })

        # Display results in a table
        if results_data:
            st.dataframe(results_data, width=800)  # Scrollable table
        else:
            st.warning("No spots processed.")
