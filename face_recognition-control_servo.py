import numpy as np
import cv2
import pyfirmata
import time

board = pyfirmata.Arduino('/dev/cu.usbserial-1410')

out_servo_x = board.get_pin('d:10:s')
out_servo_y = board.get_pin('d:11:s')
board.digital[9].write(1)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

FRAME_W = 640
FRAME_H = 480
FRAME_RATE = 5
video_capture = cv2.VideoCapture(0)
video_capture.set(3, FRAME_W)
video_capture.set(4, FRAME_H)
video_capture.set(5, FRAME_RATE)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cam_pan = 90
        cam_tilt = 90
        # Correct relative to center of image
        turn_x  = float(-(x + w/2 - (FRAME_W/2)))
        turn_y  = float(y + h/2 - (FRAME_H/2))

        # Convert to percentage offset
        turn_x  /= float(FRAME_W/2)
        turn_y  /= float(FRAME_H/2)

        # Scale offset to degrees
        turn_x   *= 10 # VFOV
        turn_y   *= 10 # HFOV
        np.clip(turn_x, -7.5, 7.5)
        np.clip(turn_y, -5.5, 5.5)
        max_turn_x = 7.5
        min_turn_x = -7.5
        max_turn_y = 5.5
        min_turn_y = -5.5
        max_servo_x = 150.0
        min_servo_x = 30.0
        max_servo_y = 150.0
        min_servo_y = 30.0
        servo_x = (((turn_x - min_turn_x) / (max_turn_x - min_turn_x)) * (max_servo_x - min_servo_x)) + min_servo_x
        servo_y = (((turn_y - min_turn_y) / (max_turn_y - min_turn_y)) * (max_servo_y - min_servo_y)) + min_servo_y
        # if(servo_x >= 150):
        #     servo_x = 150
        #     board.digital[9].write(0)
        # elif(servo_x <= 30):
        #     servo_x = 30
        #     board.digital[9].write(0)
        # else:
        #     servo_x = servo_x
        #     board.digital[9].write(0)
        # if(servo_y >= 150):
        #     servo_y = 150
        #     board.digital[9].write(0)
        # elif(servo_y <= 30):
        #     servo_y = 30
        #     board.digital[9].write(0)
        # else:
        #     servo_y = servo_y
        #     board.digital[9].write(0)
        
        out_servo_x.write(int(servo_x))
        out_servo_y.write(int(servo_y))
        print("Servo X:", int(servo_x))
        print("Servo Y:",int(servo_y))
        print("------------------")
        
        cv2.putText(video_frame, "X : " + str(float(turn_x)) + " Y: " + str(float(turn_y)), (20,20), 	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
        cv2.putText(video_frame, "Servo X : " + str(int(servo_x)) + " Servo Y: " + str(int(servo_y)), (20,50), 	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
    return faces

while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame

    cv2.imshow(
        "My Face Detection Project", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()