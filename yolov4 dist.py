import cv2 as cv

KNOWN_DISTANCE = 200
PERSON_WIDTH = 40
CAR_WIDTH = 100
MOTORBIKE_WIDTH = 40
BUS_WIDTH = 100
TRUCK_WIDTH = 100

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

class_names =  []
with open('YOLOv4\classes.txt', 'r') as f:
    class_name = f.readlines()
    for name in class_name:
        class_names.append(name. strip())

yoloNet = cv.dnn.readNet('YOLOv4\yolov4-tiny.cfg', 'YOLOv4\yolov4-tiny.weights')
model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(scale=1/255, size=(416, 416), swapRB=True)

def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        label = class_names[classid]
        cv.rectangle(image, box, (0, 255, 0), 2)
        cv.putText(image, label, (box[0], box[1]-14), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if classid == 0:
            data_list.append([class_names[classid], box[2],(box[0],box[1]-2)])
        elif classid == 2:
            data_list.append([class_names[classid], box[2],(box[0],box[1]-2)])
        elif classid == 3:
            data_list.append([class_names[classid], box[2],(box[0],box[1]-2)])
        elif classid == 5:
            data_list.append([class_names[classid], box[2],(box[0],box[1]-2)])
        elif classid == 7:
            data_list.append([class_names[classid], box[2],(box[0],box[1]-2)])
    return data_list

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

def distance_finder(focal_length, real_object_width,width_in_frame):
    distance = (real_object_width * focal_length) /width_in_frame
    return distance

ref_person = cv.imread(r'picture_test\person. jpg')
person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

ref_car = cv.imread(r'picture_test\car. jpg')
car_data = object_detector(ref_car)
car_width_in_rf = car_data[1][1]

ref_motorbike = cv.imread(r'picture_test\motorbike. jpg')
motorbike_data = object_detector(ref_motorbike)
motorbike_width_in_rf = motorbike_data[2][1]

ref_bus = cv.imread(r'picture_test\motorbike. jpg')
bus_data = object_detector(ref_motorbike)
bus_width_in_rf = bus_data[2][1]

ref_truck = cv.imread(r'picture_test\motorbike. jpg')
truck_data = object_detector(ref_truck)
truck_width_in_rf = truck_data[3][1]

print(f'Person width in pixels : {person_width_in_rf}\
        \nCar width in pixel: {car_width_in_rf}\
        \nMotorbike width in pixel: {car_width_in_rf}\
        \nBus width in pixel: {car_width_in_rf}\
        \nTruck width in pixel: {car_width_in_rf}')

focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
focal_car = focal_length_finder(KNOWN_DISTANCE, CAR_WIDTH, car_width_in_rf)
focal_motorbike = focal_length_finder(KNOWN_DISTANCE, MOTORBIKE_WIDTH, car_width_in_rf)
focal_bus = focal_length_finder(KNOWN_DISTANCE, BUS_WIDTH, car_width_in_rf)
focal_truck = focal_length_finder(KNOWN_DISTANCE, TRUCK_WIDTH, car_width_in_rf)

capture = cv.VideoCapture(r'video_test\[{uncut}| SaiGon Day Ride to District 2 Motorcycle Scooter POV Day Drive Vietnam.mp4')
while True:
    ret, frame = capture.read()
    data = object_detector(frame)

    for d in data:
        if d[0] == 'person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            distance = distance/100
            x, y = d[2]
        elif d[0] == 'car':
            distance = distance_finder (focal_car, CAR_WIDTH, d[1])
            distance = distance/100
            x, y = d[2]
        elif d[0] == 'motorbike':
            distance = distance_finder (focal_motorbike, MOTORBIKE_WIDTH, d[1])
            distance = distance/100
            x, y = d[2]
        elif d[0] == 'bus':
            distance = distance_finder (focal_bus, BUS_WIDTH, d[1])
            distance = distance/100
            x, y = d[2]
        elif d[0] == 'truck':
            distance = distance_finder (focal_truck, TRUCK_WIDTH, d[1])
            distance = distance/100
            x, y = d[2]
        cv.putText(frame, f'Distance: {round(distance, 2)} meters', (x + 5, y + 13), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv.imshow('Camera-Based Distance Estimation System', frame)
    if cv.waitKey(1) & 0xFF==27:
        break

cv.destroyAllWindows()
capture.release()