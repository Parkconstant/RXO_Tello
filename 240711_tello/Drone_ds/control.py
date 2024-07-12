import cv2
from djitellopy import tello
import cvzone

thres = 0.55
nmsThres = 0.2

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')
print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamoff()
me.streamon()


# 드론 상승
me.takeoff()
me.move_up(50)  # 예시로 50cm 상승

while True:
    img = me.get_frame_read().frame
    img_h, img_w, _ = img.shape
    center_x, center_y = img_w // 2, img_h // 2

    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
    if len(bbox) > 0:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cvzone.cornerRect(img, box)
            cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 255, 0), 2)

            if classNames[classId - 1] == "person":
                x, y, w, h = box
                cx, cy = x + w // 2, y + h // 2  # Calculate the center of the detected person

                # Calculate movement offset
                error_x = center_x - cx
                error_y = center_y - cy
                
                # Control drone to move towards the person
                speed_x = -int(error_x / 100)  # Negative sign to correct direction
                speed_y = int(error_y / 100) if abs(error_y) > 20 else 0  # Avoid minor vertical adjustments
                me.send_rc_control(speed_x, 0, speed_y, 0)
                break  # Follow only the first detected person
    else:
        me.send_rc_control(0, 0, 0, 0)  # Hover if no person is detected

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("Image", img)
    cv2.waitKey(1)

me.land()
cv2.destroyAllWindows()
