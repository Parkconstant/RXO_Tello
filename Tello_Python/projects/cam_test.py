from djitellopy import Tello
import time 
import cv2


def main():
    tello = Tello()
    tello.connect()
    cap = cv2.VideoCapture(0)

    while True :
        ret, frame = cap.read()

        cv2.imshow("Tello", frame)

        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tello.end()

if __name__ =="__main__":
    main()