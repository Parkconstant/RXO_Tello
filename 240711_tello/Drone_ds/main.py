import cv2
from djitellopy import tello
import time

# Thresholds for detection
thres = 0.55  # Detection threshold
nmsThres = 0.2  # Non-maximum suppression threshold

# Load class names
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')

# Load model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize and connect to Tello
me = tello.Tello()
me.connect()
print(f'Battery: {me.get_battery()}%')
me.streamoff()
me.streamon()
me.takeoff()
me.move_up(30)

desired_distance = 150  # Desired pixel width of the person

# Perform a 360-degree scan
print("Performing a 360-degree scan...")
for _ in range(36):  # 36 steps of 10 degrees each to complete a full circle
    me.send_rc_control(0, 0, 0, 10)  # Rotate right at a speed of 50
    time.sleep(0.2)  # Wait for 0.2 seconds
me.send_rc_control(0, 0, 0, 0)  # Stop rotation

while True:
    img = me.get_frame_read().frame
    img_h, img_w, _ = img.shape
    center_x, center_y = img_w // 2, img_h // 2

    # Object detection
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
    if len(bbox) > 0:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classNames[classId - 1] == "person":
                x, y, w, h = box
                cx, cy = x + w // 2, y + h // 2
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}%',
                            (x + 10, y + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

                # Control logic to maintain distance
                error_x = cx - center_x
                error_y = center_y - cy
                error_distance = desired_distance - w

                speed_x = int(error_x * 0.2)
                speed_y = int(error_y * 0.2)
                speed_z = int(error_distance * 0.1)

                me.send_rc_control(speed_x, speed_y, 0, speed_z)
                break  # Only follow one person
    else:
        me.send_rc_control(0, 0, 0, 0)  # Hover in place if no person is detected

    # Display the image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

me.land()
cv2.destroyAllWindows()