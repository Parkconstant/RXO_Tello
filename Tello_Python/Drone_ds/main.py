# # import cv2
# # from djitellopy import tello
# # import cvzone

# # thres = 0.55  # 객체 탐지 임계값
# # nmsThres = 0.2  # Non-maximum suppression 임계값

# # classNames = []  # 클래스 이름을 저장할 리스트
# # classFile = 'coco.names'  # 클래스 이름 파일
# # with open(classFile, 'rt') as f:
# #     classNames = f.read().split('\n')

# # configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # 모델 설정 파일 경로
# # weightsPath = "frozen_inference_graph.pb"  # 가중치 파일 경로

# # net = cv2.dnn_DetectionModel(weightsPath, configPath)
# # net.setInputSize(320, 320)
# # net.setInputScale(1.0 / 127.5)
# # net.setInputMean((127.5, 127.5, 127.5))
# # net.setInputSwapRB(True)

# # me = tello.Tello()
# # me.connect()
# # print(me.get_battery())
# # me.streamoff()
# # me.streamon()

# # me.takeoff()
# # me.move_up(30)  # 초기 상승 명령, 드론이 일정 높이로 상승

# # while True:
# #     img = me.get_frame_read().frame
# #     classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
# #     if len(bbox) > 0:  # 객체가 감지된 경우
# #         try:
# #             for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
# #                 cvzone.cornerRect(img, box)
# #                 cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
# #                             (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
# #                             1, (0, 255, 0), 2)
# #                 # 여기에서 특정 객체(예: 사람) 감지 시 호버링 또는 추가 동작을 구현할 수 있음
# #         except:
# #             pass
# #     # 객체 감지 여부와 관계없이 send_rc_control(0, 0, 0, 0) 명령을 통해 명시적으로 호버링 상태를 유지
# #     me.send_rc_control(0, 0, 0, 0)

# #     if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 반복문 탈출
# #         break

# #     cv2.imshow("Image", img)
# #     cv2.waitKey(1)

# # # 반복문 탈출 후 드론 착륙 및 자원 해제
# # me.land()
# # cv2.destroyAllWindows()


# import cv2
# from djitellopy import tello
# import cvzone

# thres = 0.55  # 객체 탐지 임계값
# nmsThres = 0.2  # Non-maximum suppression 임계값

# classNames = []  # 클래스 이름을 저장할 리스트
# classFile = 'coco.names'  # 클래스 이름 파일
# with open(classFile, 'rt') as f:
#     classNames = f.read().split('\n')

# configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # 모델 설정 파일 경로
# weightsPath = "frozen_inference_graph.pb"  # 가중치 파일 경로

# net = cv2.dnn_DetectionModel(weightsPath, configPath)
# net.setInputSize(320, 320)
# net.setInputScale(1.0 / 127.5)
# net.setInputMean((127.5, 127.5, 127.5))
# net.setInputSwapRB(True)

# me = tello.Tello()
# me.connect()
# print(me.get_battery())
# me.streamoff()
# me.streamon()

# me.takeoff()
# me.move_up(30)  # 초기 상승 명령, 드론이 일정 높이로 상승

# desired_distance = 150  # 목표 거리 (픽셀 단위)

# while True:
#     img = me.get_frame_read().frame
#     img_h, img_w, _ = img.shape
#     center_x, center_y = img_w // 2, img_h // 2

#     classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)
#     if len(bbox) > 0:
#         for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
#             if classNames[classId - 1] == "person":
#                 x, y, w, h = box
#                 cx, cy = x + (w // 2), y + (h // 2)  # 객체의 중심 좌표
#                 cvzone.cornerRect(img, box)
#                 cv2.putText(img, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
#                             (x + 10, y + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                             1, (0, 255, 0), 2)

#                 # 드론 위치 조정 로직
#                 error_x = center_x - cx
#                 error_y = center_y - cy
#                 error_distance = desired_distance - w  # 가로 폭을 거리 추정에 사용

#                 # PID 컨트롤러를 사용하여 드론의 움직임을 미세 조정할 수 있습니다.
#                 # 여기에서는 간단한 비례 제어만을 사용
#                 speed_x = int(error_x * 0.2)
#                 speed_y = int(error_y * 0.2)
#                 speed_z = int(error_distance * 0.1)

#                 me.send_rc_control(speed_x, speed_y, 0, speed_z)

#     else:
#         me.send_rc_control(0, 0, 0, 0)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     cv2.imshow("Image", img)
#     cv2.waitKey(1)

# me.land()
# cv2.destroyAllWindows()





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
                error_x = center_x - cx
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
