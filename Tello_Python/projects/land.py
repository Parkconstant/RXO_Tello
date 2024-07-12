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

if __name__ == "__main__":
    print("Start program....")
    drone_control()
    print("\n\nEnd of Program\n")
