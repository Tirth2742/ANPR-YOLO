import streamlit as st
import cv2
import time
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import easyocr
import tempfile

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Load models
coco_model = YOLO('yolov8n.pt')  # Replace with your vehicle detection model
license_plate_detector = YOLO('license_plate_detector.pt')  # Replace with your license plate detection model

# Define vehicle classes (COCO class IDs for car, bus, truck, motorcycle)
vehicles = [2, 3, 5, 7]

# Streamlit app layout
st.title("License Plate Detection with EasyOCR")
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Data storage for detected plates
detected_plates = []

# Display the table at the start (empty table)
table_placeholder = st.empty()

# Create a placeholder dataframe for the table with separate Date and Time columns
df = pd.DataFrame(columns=['License Plate', 'Date', 'Time'])

if uploaded_video is not None:
    # Create a temporary file to store the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    # Start video feed from the uploaded video
    cap = cv2.VideoCapture(tfile.name)

    # Set up the video container in Streamlit
    video_placeholder = st.empty()

    # Read frames and process
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Finished processing the video")
            break

        # Detect vehicles
        detections = coco_model(frame)[0]
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                # Draw vehicle bounding box (optional)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Detect license plates
                license_plates = license_plate_detector(frame)[0]
                for license_plate in license_plates.boxes.data.tolist():
                    lx1, ly1, lx2, ly2, lscore, lclass_id = license_plate

                    # Draw license plate bounding box
                    cv2.rectangle(frame, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (255, 0, 0), 2)

                    # Crop the license plate region
                    license_plate_crop = frame[int(ly1):int(ly2), int(lx1):int(lx2)]

                    # Apply EasyOCR to the cropped license plate
                    license_plate_text = reader.readtext(license_plate_crop, detail=0)

                    # If text is detected
                    if license_plate_text:
                        # Get the current time and date separately
                        now = datetime.now()
                        current_date = now.strftime('%Y-%m-%d')
                        current_time = now.strftime('%H:%M:%S')

                        # Save the detected license plate, date, and time
                        detected_plates.append({
                            'License Plate': license_plate_text[0],
                            'Date': current_date,
                            'Time': current_time
                        })

                        # Limit the number of rows displayed
                        if len(detected_plates) > 10:
                            detected_plates = detected_plates[-10:]

                        # Update the dataframe with new data
                        df = pd.DataFrame(detected_plates)

                        # Update the table in the Streamlit app
                        table_placeholder.table(df)

        # Convert the frame from BGR to RGB (Streamlit uses RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the video in Streamlit
        video_placeholder.image(frame_rgb)

    # Release video capture when done
    cap.release()
