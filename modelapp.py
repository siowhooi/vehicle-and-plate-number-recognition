import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image

# Load YOLO model
model = YOLO(r"best.pt")  # Ensure 'best.pt' is in the same directory or update the path

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

            return plate_image, ''.join(plate_text)

    return None, "No license plate detected"

# Streamlit app
st.title("License Plate Recognition")
st.sidebar.title("Options")

# Image upload or webcam capture
option = st.sidebar.radio("Choose Input Method:", ("Upload an Image", "Use Webcam"))

if option == "Upload an Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image
        with st.spinner("Processing..."):
            plate_image, plate_text = process_image(image)

        # Display results
        if plate_image is not None:
            st.image(plate_image, caption="Detected License Plate", use_column_width=True)
            st.success(f"Recognized Plate Number: {plate_text}")
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
                plate_image, plate_text = process_image(image)

            # Display results
            if plate_image is not None:
                st.image(plate_image, caption="Detected License Plate", use_column_width=True)
                st.success(f"Recognized Plate Number: {plate_text}")
            else:
                st.warning("No license plate detected in the image.")
        else:
            st.error("Failed to capture an image. Please try again.")

st.sidebar.write("Developed with YOLO and EasyOCR")
