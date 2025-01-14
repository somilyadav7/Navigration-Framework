import cv2
import mediapipe as mp
from ultralytics import YOLO
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(message):
    """Speak the given message."""
    engine.say(message)
    engine.runAndWait()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize YOLO model
model = YOLO("yolov8n.pt")  # Replace with your YOLO model

# Dictionary to track bounding box sizes over time
bbox_history = {}

def is_person_approaching(id, current_bbox):
    """
    Determine if a person is approaching based on bounding box size over time.
    :param id: Unique ID for the detected person.
    :param current_bbox: Current bounding box dimensions (x1, y1, x2, y2).
    :return: Boolean indicating if the person is approaching.
    """
    global bbox_history

    # Calculate the bounding box size (area)
    x1, y1, x2, y2 = current_bbox
    area = (x2 - x1) * (y2 - y1)

    # Update history for this ID
    if id not in bbox_history:
        bbox_history[id] = []
    bbox_history[id].append(area)

    # Keep the history short
    if len(bbox_history[id]) > 5:
        bbox_history[id].pop(0)

    # Check if the area consistently increases
    if len(bbox_history[id]) > 1 and all(
        bbox_history[id][i] < bbox_history[id][i + 1] for i in range(len(bbox_history[id]) - 1)
    ):
        return True

    return False

def analyze_pose(landmarks):
    """
    Analyze pose landmarks to detect potential threats or possible threats.
    :param landmarks: Normalized landmarks from MediaPipe Pose.
    :return: Tuple (possible_threat, confirmed_threat).
    """
    if not landmarks:
        return False, False

    # Retrieve wrist and nose coordinates
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    # Possible threat: One or both hands raised above the nose
    possible_threat = (left_wrist.y < nose.y) or (right_wrist.y < nose.y)

    # Confirmed threat: Raised hands and close to the camera
    confirmed_threat = possible_threat and (
        abs(left_wrist.z) < 0.15 or abs(right_wrist.z) < 0.15
    )

    return possible_threat, confirmed_threat

def detect_threat(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % int(fps // 2) != 0:  # Process every 2nd frame to improve speed
            continue

        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (640, 360))

        # Use YOLO to detect people
        results = model(frame_resized)
        detections = results[0].boxes

        for i, box in enumerate(detections):
            # Only process detections labeled as "person"
            if box.cls[0] != 0:  # Class 0 is "person" in COCO dataset
                continue

            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Check if the person is approaching
            if is_person_approaching(i, (x1, y1, x2, y2)):
                speak("Person approaching")

            # Crop the detected person
            person_crop = frame_resized[y1:y2, x1:x2]

            # Convert to RGB for MediaPipe
            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

            # Perform pose estimation
            result = pose.process(person_rgb)

            if result.pose_landmarks:
                # Analyze the pose for potential threats
                possible_threat, confirmed_threat = analyze_pose(result.pose_landmarks.landmark)

                if confirmed_threat:
                    # Draw bounding box and alert for confirmed threat
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    speak("Threat detected! Please be cautious!")
                elif possible_threat:
                    # Draw bounding box and alert for possible threat
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    speak("Possible threat, be cautious")
                else:
                    # Draw bounding box in green for safe individuals
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Threat Detection", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run threat detection on a video
detect_threat("knife1.mp4")
