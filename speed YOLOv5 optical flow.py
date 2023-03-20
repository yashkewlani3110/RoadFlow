import cv2
import torch
import numpy as np
import sys
sys.path.append('/Users/yashkewlani/Documents/Roadflow/yolov8_tracking')
from trackers.strongsort.sort.tracker import StrongSort
from trackers.strongsort.utils.parser import get_config
from trackers.strongsort.utils.draw import draw_boxes
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# Load YOLOv5 model
device = select_device()
model = attempt_load('yolov5s.pt', map_location=device)
model.conf = 0.4
model.iou = 0.5
model.classes = None
model.names = model.module.names if hasattr(model, 'module') else model.names

# Initialize DeepSORT
cfg = get_config()
cfg.merge_from_file('yolov8_tracking/configs/strongsort.yaml')
strongsort = StrongSort(cfg.STRONGSORT.REID_CKPT, max_dist=cfg.STRONGSORT.MAX_DIST, min_confidence=cfg.STRONGSORT.MIN_CONFIDENCE, nms_max_overlap=cfg.STRONGSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE, max_age=cfg.STRONGSORT.MAX_AGE, n_init=cfg.STRONGSORT.N_INIT, nn_budget=cfg.STRONGSORT.NN_BUDGET, use_cuda=True)

# Initialize video capture
cap = cv2.VideoCapture(0)
_, frame = cap.read()
height, width, _ = frame.shape

# Parameters for optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0, 255, (1000, 3))
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for YOLOv5
    img = torch.from_numpy(frame.transpose(2, 0, 1)).to(device).float() / 255.0
    img = img.unsqueeze(0)
    img = img[..., :height, :width]

    # Run YOLOv5 and apply non-max suppression
    pred = model(img)[0]
    pred = non_max_suppression(pred, model.conf, model.iou, classes=model.classes, agnostic=False)
    bboxes = []

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            # Extract bounding boxes for vehicles (car: 2, truck: 7)
            for *xyxy, conf, cls in det:
                if int(cls) in [2, 7]:
                    x1, y1, x2, y2 = xyxy
                    bboxes.append([x1, y1, x2, y2, conf, conf, cls])

    # Apply DeepSORT
    outputs = strongsort.update(bboxes, frame)

    # Draw tracking information
    if len(outputs) > 0:
        bbox_xywh = []
        identities = []
        for x1, y1, x2, y2, track_id in outputs:
            bbox_xywh.append([x1, y1, x2 - x1, y2 - y1])
            identities.append(track_id)
        frame = draw_boxes(frame, bbox_xywh, identities)

    # Calculate optical flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **lk_params)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
    if p1 is not None:
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            frame = cv2.line(frame, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

    # Display the result
    cv2.imshow("frame", frame)

    # Update previous frame and points
    prev_gray = gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
