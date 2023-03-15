# Import libraries
import cv2
import numpy as np
import easyocr
from deep_sort import DeepSort

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize DeepSort tracker
tracker = DeepSort()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Define license plate class id (depends on your dataset)
lp_class_id = 2

# Define video source (can be a file or a camera)
video_source = 'video.mp4'

# Open video capture
cap = cv2.VideoCapture(video_source)

# Loop over frames
while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB format for YOLOv5 model
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run inference on frame with YOLOv5 model
    results = model(rgb_frame)

    # Get detections from results (xyxy format)
    detections = results.xyxy[0]

    # Filter out detections that are not license plates
    lp_detections = detections[detections[:, -1] == lp_class_id]

    # Convert lp_detections to xywh format for DeepSort tracker
    lp_detections_xywh = lp_detections[:, :4].clone()
    lp_detections_xywh[:, 0] = (lp_detections[:, 0] + lp_detections[:, 2]) / 2 # x center
    lp_detections_xywh[:, 1] = (lp_detections[:, 1] + lp_detections[:, 3]) / 2 # y center
    lp_detections_xywh[:, 2] = lp_detections[:, 2] - lp_detections[:, 0] # width 
    lp_detections_xywh[:, 3] = lp_detections[:, 3] - lp_detections[:, 1] # height

    # Get confidences from detections 
    confidences = lp_detections[:,-2]

    # Update tracker with new detections 
    tracks, _ = tracker.update(lp_detections_xywh.cpu().numpy(), confidences.cpu().numpy(), rgb_frame)

    # Loop over tracks 
    for track in tracks:
       # Get track id 
       track_id = track.track_id

       # Get bounding box coordinates (xyxy format)
       x1, y1, x2, y2 = track.to_tlbr().astype(int)

       # Crop license plate region from frame 
       lp_region = frame[y1:y2, x1:x2]

       # Read license plate characters with EasyOCR 
       result = reader.readtext(lp_region)

       # Get license plate text and confidence 
       if result:
           text, confidence = result[0][1], result[0][2]
       
           # Draw bounding box and text on frame 
           cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
           cv2.putText(frame,f'{text} ({confidence:.2f})', (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX ,0.8,(255,255,255),3,cv.LINE_AA) 

    # Show frame 
    cv2.imshow('frame',frame)

    # Wait for key press 
    key=cv2.waitKey(30) & 0xFF

    if key==27: break

cap.release()
cv2.destroyAllWindows()