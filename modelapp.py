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

            return plate_image, ''.join(plate_text), class_name

    return None, "No license plate detected", None

# Streamlit app
st.title("License Plate Recognition System")
st.sidebar.title("Menu")

# Sidebar menu
menu_option = st.sidebar.radio("Choose Menu:", ("Spot Management", "Results"))

# Menu 1: Spot Management
if menu_option == "Spot Management":
    st.subheader("Spot Management: Upload or Capture Images")
    
    # Four spots (2x2 layout)
    cols = st.columns(2)

    spot_images = [None] * 4  # Placeholder for spot images
    spot_results = []

    for i, col in enumerate(cols * 2):  # Create 4 columns for 4 spots
        with col:
            st.markdown(f"### Spot {i+1}")
            upload_option = st.radio(
                f"Upload or Capture for Spot {i+1}",
                ("Upload Image", "Use Webcam"),
                key=f"option_{i+1}"
            )
            
            if upload_option == "Upload Image":
                uploaded_file = st.file_uploader(
                    f"Upload Image for Spot {i+1}", type=["jpg", "jpeg", "png"], key=f"upload_{i+1}"
                )
                if uploaded_file:
                    image = Image.open(uploaded_file)
                    spot_images[i] = image
                    st.image(image, caption=f"Spot {i+1} Image", use_column_width=True)
            elif upload_option == "Use Webcam":
                run = st.button(f"Capture Image for Spot {i+1}", key=f"capture_{i+1}")
                if run:
                    cap = cv2.VideoCapture(0)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        spot_images[i] = image
                        st.image(image, caption=f"Spot {i+1} Captured Image", use_column_width=True)
                    else:
                        st.error("Failed to capture an image. Please try again.")

    # Process all images for license plate recognition
    for i, image in enumerate(spot_images):
        if image:
            with st.spinner(f"Processing Spot {i+1}..."):
                plate_image, plate_text, vehicle_class = process_image(image)
            if plate_image is not None:
                spot_results.append(
                    {
                        "spot": i + 1,
                        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "vehicle_class": vehicle_class,
                        "plate_number": plate_text,
                    }
                )
                st.image(plate_image, caption=f"Detected License Plate (Spot {i+1})", use_column_width=True)
                st.success(f"Spot {i+1} - Plate Number: {plate_text}")
            else:
                st.warning(f"No license plate detected in Spot {i+1}")

# Menu 2: Results
elif menu_option == "Results":
    st.subheader("Detected Results")
    # Placeholder for results
    if "spot_results" not in st.session_state:
        st.session_state.spot_results = []

    if spot_results:
        st.session_state.spot_results.extend(spot_results)

    if st.session_state.spot_results:
        st.table(st.session_state.spot_results)
    else:
        st.info("No results to display yet.")
