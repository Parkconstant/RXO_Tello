import cv2 # OpenCV라이브러리, 이미지 및 비디오 처리 기능 
import numpy as np # 수학 연산 라이브러리
from dynamikontrol import Module 


CONFIDENCE = 0.9 # 객체탐지 신뢰도 임계값, 이 값보다 높은 신뢰도를 가진 객체만 처리
THRESHOLD = 0.3 #Non-Maximum Suppression(NMS)에 사용되는 임계값
# NMS란? 객체 탐지에서 중복된 박스를 제거하는 기술, 즉 차량번호판 인식에서 여러개의 번호판 박스가 겹쳐서 나타날 수 있는데 NMS를 적용하여 겹친 번호판 박스중에서 가장 높은 정확도를 가진 박스를 선택하고
# 나머지는 제거하여 정확한  번호판을 인식, 그렇다면 0.3 미만의 박스들은 우선적으로 걸러지고 그 후 0.3이상들의 박스들에서 가장 높은 값을 선택하는 건가? -> 그렇다고 한다.
LABELS = ['Car', 'Plate'] #탐지한 객체의 라벨, Car와 Plate(번호판)으로 분류
CAR_WIDTH_TRESHOLD = 300 #차량 폭 임계값, 이값보다 커야 모터가 작동함. 하지만 현재 카메라 근처로 차량을 움직였을 시 300중후반대가 나옴. 따라서 모터 작동 X 임계값을 낮춰야 할 것 같음.

cap = cv2.VideoCapture(0) # 장치관리자를 확인해보면 내가 연결한 카메라가 1번자리에 있는데 1번자리는 노트북 카메라로 인식한다. 따라서 0번으로 해야 됨.
cfg_path = 'C:/Users/Constant/Desktop/Parking/Parking Gate-Yolov4/cfg/yolov4-ANPR.cfg'
weights_path = 'C:/Users/Constant/Desktop/Parking/Parking Gate-Yolov4/yolov4-ANPR.weights'
net = cv2.dnn.readNetFromDarknet('cfg_path', 'weight_path') # YOLOv4 모델을 Darknet 형식에서 로드, cfg와 weights 파일이 필요
# YOLOv4란 무엇인가? -> 객체 검출 알고리즘, 이미지를 한번만 보고 여러 객체의 위치와 종류를 동시에 예측하는 방법
# Darknet이란? 오픈소스 신경망 프레임워크
# cfg, weights 파일의 역할은 무엇인가?

# cfg 파일은 네트워크 모델의 구조와 설정을 정의하는 파일
# YOLO 모델의 cfg 파일은 네트워크의 층(layer) 구조, 필터(filter) 크기, 활성화 함수 등을 설정

# weights 파일은 신경망의 학습된 가중치(weights)를 저장하는 파일
# 네트워크 모델은 초기화된 상태에서 학습 데이터를 통해 학습을 거치며, 이 과정에서 가중치가 조정
# 학습이 완료된 후 가중치가 최적화 된 상태로 저장
# 이 파일은 네트워크가 이미지나 비디오에서 실제 객체를 탐지할 수 있도록 합니다. 학습된 가중치는 네트워크가 객체를 인식하는 데 중요한 역할을 합니다.


module = Module() # dynamikontrol 모듈 초기화, 이 모듈은 외부 모터를 제어할 기능을 제공

while cap.isOpened(): #비디오 프레임 처리 및 객체 탐지
    ret, img = cap.read() # 카메라로 부터 프레임을 읽어옴, ret은 읽기가 성공한지를 확인하기 위한 boolean값, img실제 프레임
    if not ret:
        break

    H, W, _ = img.shape # 높이,너비, 채널수(_)추출

    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255., size=(416, 416), swapRB=True) # 이미지를 YOLOv4 모델이 입력으로 사용할 수 있는 형식으로 변환, 즉 신경망에 넣을 수 있는 형식으로 변환
    # scalefactor=1/255. 이미지 픽셀 값을 0에서 1사이의 값으로 정규화
    # size=(416, 416)은 신경망의 입력 이미지 크기를 지정
    # swapRB=True는 OpenCV의 BGR 형식을 RGB로 전환, 왜 OpenCV는 BGR 방식으로 설계해서 사람을 귀찮게 하는가? -> 개발 당시 저수준의 이미지 처리 루틴에서는 BGR방식이 더 효율적이었을 수 도 있다는 추측.
    net.setInput(blob)  #  전처리된 이미지를 신경망에 입력으로 설정
    output = net.forward() # net.forward() 입력 이미지에 대한 순방향 전파를 수행, 입력층 -> 은닉층 -> 출력층 
    # output 객체 검출 네트워크의 출력. 각 객체에 대해 여러 개의 박스와 해당 클래스에 대한 점수가 포함

    boxes, confidences, class_ids = [], [], []

    for det in output:
        box = det[:4] # 각 객체의 경계 상자 좌표 
        scores = det[5:] # 객체 클래스의 점수
        class_id = np.argmax(scores) # 가장 높은 점수를 가진 클래스의 인덱스
        confidence = scores[class_id] # 해당 클래스의 점수

        if confidence > CONFIDENCE: # 임계값보다 높은 신뢰도를 가진 객체만을 처리
            cx, cy, w, h = box * np.array([W, H, W, H]) # 상대 좌표를 절대 좌표로 변환
            x = cx - (w / 2) 
            y = cy - (h / 2) # 경계 상자의 좌측 상단 모서리의 좌표를 계산

            boxes.append([int(x), int(y), int(w), int(h)]) 
            confidences.append(float(confidence)) # 각 객체에 대한 경계 상자 좌표, 신뢰도, 클래스 ID를 리스트에 저장
            class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)
    # cv2.dnn.NMSBoxes() 함수는 Non-Maximum Suppression을 적용하여 겹치는 경계 상자를 제거하고, 신뢰도가 낮은 객체를 필터링

    if len(idxs) > 0: # NMS를 통해 선택된 객체가 하나 이상일 때 실행
        for i in idxs.flatten(): # 배열을 평평하게 하여 반복
            x, y, w, h = boxes[i] # boxes에서 인덱스 i에 해당하는 객체의 경계 상자 좌표를 가져옴

            cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2) #img에 검출된 객체의 경계 상자를 그림
            cv2.putText(img, text='%s %.2f %d' % (LABELS[class_ids[i]], confidences[i], w), org=(x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            # 객체의 클래스 이름, 신뢰도 및 너비 정보를 텍스트로 이미지에 표시
            if class_ids[i] == 0: # 객체의 클래스가 'Car'인 경우
                if w > CAR_WIDTH_TRESHOLD: # 위에 설정한 값보다 크면 모터 80도 회전
                    module.motor.angle(80)
                else: # 0도 설정
                    module.motor.angle(0)
    else: # 선택된 객체가 없는 경우
        module.motor.angle(0) # 0도

    cv2.imshow('result', img) # 최종 처리된 이미지를 result 창에 출력
    if cv2.waitKey(1) == ord('q'): # 'q'키를 누르면 무한 루프 탈출
        break
