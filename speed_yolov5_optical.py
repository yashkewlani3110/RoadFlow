import cv2
import torch
import numpy as np
from sort import Sort
from scipy.spatial import distance

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='yolov5s.pt')
model.conf = 0.5  # Confidence threshold
model.iou = 0.5  # NMS IoU threshold
model.classes = [2, 5, 7]  # Set class filter (2: car, 5: bus, 7: truck)

# Initialize SORT tracker
tracker = Sort(use_dlib=True)

# Initialize optical flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open video device")
    exit()

_, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_points = None
prev_detections = []

while True:
    # Read the camera frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects using YOLOv5
    results = model(frame)

    # Process detections
    detections = []
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        detections.append([x1, y1, x2, y2, conf])

    # Track objects using SORT
    if len(detections) > 0:
        tracked_objects = tracker.update(np.array(detections))
        prev_detections = detections
    else:
        tracked_objects = tracker.update(np.array(prev_detections))

    # Calculate optical flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_points is None:
        prev_points = np.array([[[float(x1), float(y1)]] for x1, y1, _, _, _ in tracked_objects], dtype=np.float32)
    else:
        curr_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)

        # Calculate relative speeds
        relative_speeds = []
        for i, (new, old) in enumerate(zip(curr_points, prev_points)):
            if status[i]:
                x1, y1 = new.ravel()
                x0, y0 = old.ravel()
                speed = distance.euclidean((x0, y0), (x1, y1))
                relative_speeds.append(speed)

        # Update previous frame and points
        prev_gray = gray.copy()
        prev_points = curr_points.copy()

    # Draw tracked objects and relative speeds
    for i, (x1, y1, x2, y2, _, _) in enumerate(tracked_objects):
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {i}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        if prev_points is not None and i < len(relative_speeds):
            cv2.putText(frame, f"Speed: {relative_speeds[i]:.2f}", (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Vehicle Detection and Tracking', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
