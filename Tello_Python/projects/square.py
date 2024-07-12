import threading
import cv2
import time
from djitellopy import Tello

def draw_square(drone, distance):
    for _ in range(4):
        drone.move_forward(distance)
        time.sleep(3)
        drone.rotate_clockwise(90)
        time.sleep(3)

def drone_control():
    tello = Tello()
    tello.connect()
    tello.takeoff()
    draw_square(tello, 300)
    tello.land()
    tello.end()

def webcam_capture():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Start program....")

    # 쓰레드 생성 및 실행
    drone_thread = threading.Thread(target=drone_control)
    webcam_thread = threading.Thread(target=webcam_capture)
    drone_thread.start()
    webcam_thread.start()

    # 쓰레드 종료 대기
    drone_thread.join()
    webcam_thread.join()

    print("\n\nEnd of Program\n")
