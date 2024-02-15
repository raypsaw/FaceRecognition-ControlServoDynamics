# Import all library to run this program
import numpy as np
import cv2
import pyfirmata
import time

# Initilaize board arduino (name port)
board = pyfirmata.Arduino('/dev/cu.usbserial-1410')

# Initialize pin board arduino for servo and relay
out_servo_x = board.get_pin('d:10:s')
out_servo_y = board.get_pin('d:11:s')
board.digital[9].write(1)

# Setup the pin of servo and initialize variable for servo
out_servo_x.write(90)
out_servo_y.write(90)
prev_servo_x = 90.0
prev_servo_y = 90.0
servo_x = 90.0
servo_y = 90.0
prev_error_x = 90.0
prev_error_y = 90.0
error_x = 90.0
error_y = 90.0

# Initialize gain for system control servo_x and servo_y using PID Controller
p_gain_x = 0.04
i_gain_x = 0.00001
d_gain_x = 0.005
p_gain_y = 0.048
i_gain_y = 0.00003
d_gain_y = 0.005

# Initialize for integral and derrivative variabel for PID Controller
integral_x = 0.0
prev_error_x = 0.0
integral_y = 0.0
prev_error_y = 0.0

# Import the library for face detection
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Set up frame for display the output camera
FRAME_W = 640
FRAME_H = 480
FRAME_RATE = 5
video_capture = cv2.VideoCapture(0)
video_capture.set(3, FRAME_W)
video_capture.set(4, FRAME_H)
video_capture.set(5, FRAME_RATE)

# Main Loop for this program
while True:
    # Run the camera
    result, video_frame = video_capture.read()
    if result is False:
        break

    # Face detection and create box
    gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        # Create box from each point in coordinate x,y
        cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        turn_x = float(-(x + w / 2 - (FRAME_W / 2)))
        turn_y = float(y + h / 2 - (FRAME_H / 2))

        turn_x /= float(FRAME_W / 2)
        turn_y /= float(FRAME_H / 2)

        turn_x *= 10
        turn_y *= 10
        np.clip(turn_x, -7.5, 7.5)
        np.clip(turn_y, -5.5, 5.5)

        # Initialize each variable for maximum and minimum
        max_x = 400
        min_x = 0
        max_y = 0
        min_y = 250
        max_servo_x = 30.0
        min_servo_x = 150.0
        max_servo_y = 30.0
        min_servo_y = 150.0
        max_h = 270.0
        min_h = 100.0
        max_distance = 20.0
        min_distance = 200.0

        # Formula of Linear Regression for mapping variable
        distance = (((h - min_h) / (max_h - min_h)) * (max_distance - min_distance)) + min_distance

        # Undetection if the distance so far
        if (distance > 150):
            servo_x = prev_servo_x
            servo_y = prev_servo_y
            out_servo_x.write(servo_x)
            out_servo_y.write(servo_y)
        
        # Detection face from the library
        else:
            if (x <= 198):
                error_x = 198 - x
                integral_x = integral_x + error_x
                derivative_x = error_x - prev_error_x
                p_term_x = p_gain_x * error_x
                i_term_x = i_gain_x * integral_x
                d_term_x = d_gain_x * derivative_x
                servo_x = prev_servo_x + (p_term_x + i_term_x + d_term_x)
            elif (x >= 202):
                error_x = x - 202
                integral_x = integral_x + error_x
                derivative_x = error_x - prev_error_x
                p_term_x = p_gain_x * error_x
                i_term_x = i_gain_x * integral_x
                d_term_x = d_gain_x * derivative_x
                servo_x = prev_servo_x - (p_term_x + i_term_x + d_term_x)
            else:
                servo_x = servo_x

            if (y <= 123):
                error_y = 123 - y
                integral_y = integral_y + error_y
                derivative_y = error_y - prev_error_y
                p_term_y = p_gain_y * error_y
                i_term_y = i_gain_y * integral_y
                d_term_y = d_gain_y * derivative_y
                servo_y = prev_servo_y - (p_term_y + i_term_y + d_term_y)
            elif (y >= 128):
                error_y = y - 128
                integral_y = integral_y + error_y
                derivative_y = error_y - prev_error_y
                p_term_y = p_gain_y * error_y
                i_term_y = i_gain_y * integral_y
                d_term_y = d_gain_y * derivative_y
                servo_y = prev_servo_y + (p_term_y + i_term_y + d_term_y)
            else:
                servo_y = servo_y

            if (servo_x < max_servo_x):
                servo_x = max_servo_x
            elif (servo_x > min_servo_x):
                servo_x = min_servo_x
            else:
                servo_x = servo_x

            if (servo_y < max_servo_y):
                servo_y = max_servo_y
            elif (servo_y > min_servo_y):
                servo_y = min_servo_y
            else:
                servo_y = servo_y

            # Overwrite value of servo_x and servo_y from calculation above
            out_servo_x.write(servo_x)
            out_servo_y.write(servo_y)

            # Save the previous value for calculation
            prev_error_x = error_x
            prev_error_y = error_y
            prev_servo_x = servo_x
            prev_servo_y = servo_y

        # Display each value for correction
        print("X : ", x)
        print("Y : ", y)
        print("Servo X:", int(servo_x))
        print("Servo Y:", int(servo_y))
        print("Distance:", int(distance))
        print("------------------")
        break

    cv2.imshow("My Face Detection Project", video_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()