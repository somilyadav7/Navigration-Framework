from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO('static/models/money.pt')  # Path to the exported model

# Load the video
input_video_path = 'test1.mp4'  # Replace with the path to your video
output_video_path = 'video.mp4'  # Path to save the annotated video

# OpenCV video capture
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video writer
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()