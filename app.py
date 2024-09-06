import streamlit as st
import cv2
import time
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import easyocr

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Load models
coco_model = YOLO('yolov8n.pt')  # Replace with your vehicle detection model
license_plate_detector = YOLO('license_plate_detector.pt')  # Replace with your license plate detection model

# Define vehicle classes (COCO class IDs for car, bus, truck, motorcycle)
vehicles = [2, 3, 5, 7]

# Streamlit app layout
st.title("Live License Plate Detection with EasyOCR")
run = st.checkbox('Run License Plate Detection')

# Start video feed
video_feed = st.empty()

cap = cv2.VideoCapture(0)  # Use webcam

# Reduce frame size for faster processing
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Frame rate control
fps_limit = 10  # Limit to 10 FPS to make the stream smoother
prev_time = 0

# Data storage for detected plates
detected_plates = []

# Display the table at the start (empty table)
table_placeholder = st.empty()

# Create a placeholder dataframe for the table with separate Date and Time columns
df = pd.DataFrame(columns=['License Plate', 'Date', 'Time'])

# Read frames and process
while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Unable to capture video")
        break

    # FPS limit
    current_time = time.time()
    if current_time - prev_time < 1.0 / fps_limit:
        continue
    prev_time = current_time

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

    # Display the video feed in Streamlit
    video_feed.image(frame, channels="BGR")

# Release video capture when done
cap.release()
