import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# ================= GPIO =================
GPIO.setmode(GPIO.BCM)

IN1, IN2, IN3, IN4 = 17, 27, 22, 23
ENA, ENB = 18, 19
RELAY = 24
TRIG, ECHO = 5, 6

GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB, TRIG], GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# 🔥 FLOAT relay OFF
GPIO.setup(RELAY, GPIO.IN)

pwm_left = GPIO.PWM(ENA, 1000)
pwm_right = GPIO.PWM(ENB, 1000)
pwm_left.start(0)
pwm_right.start(0)

speed = 40
AUTO_MODE = False
SPRAY_ENABLED = True

# ================= MOTOR =================
def stop():
    GPIO.output([IN1, IN2, IN3, IN4], 0)
    pwm_left.ChangeDutyCycle(0)
    pwm_right.ChangeDutyCycle(0)

def forward():
    GPIO.output(IN1,1); GPIO.output(IN2,0)
    GPIO.output(IN3,1); GPIO.output(IN4,0)
    pwm_left.ChangeDutyCycle(speed)
    pwm_right.ChangeDutyCycle(speed)

def backward():
    GPIO.output(IN1,0); GPIO.output(IN2,1)
    GPIO.output(IN3,0); GPIO.output(IN4,1)
    pwm_left.ChangeDutyCycle(speed)
    pwm_right.ChangeDutyCycle(speed)

def left_turn():
    GPIO.output(IN1,0); GPIO.output(IN2,1)
    GPIO.output(IN3,1); GPIO.output(IN4,0)
    pwm_left.ChangeDutyCycle(speed * 0.6)
    pwm_right.ChangeDutyCycle(speed)

def right_turn():
    GPIO.output(IN1,1); GPIO.output(IN2,0)
    GPIO.output(IN3,0); GPIO.output(IN4,1)
    pwm_left.ChangeDutyCycle(speed)
    pwm_right.ChangeDutyCycle(speed * 0.6)

# ================= RELAY (FLOAT) =================
def relay_on():
    GPIO.setup(RELAY, GPIO.OUT)
    GPIO.output(RELAY, GPIO.LOW)

def relay_off():
    GPIO.setup(RELAY, GPIO.IN)

# ================= ULTRASONIC =================
def get_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.0002)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start_time = time.time()

    while GPIO.input(ECHO) == 0:
        if time.time() - start_time > 0.02:
            return None

    pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        if time.time() - pulse_start > 0.02:
            return None

    pulse_end = time.time()
    distance = (pulse_end - pulse_start) * 17150
    return distance

# ================= GAMMA =================
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0)**invGamma)*255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# ================= AI VISION =================
def adaptive_vision(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(3.0, (8,8))
    l = clahe.apply(l)

    frame = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

    brightness = np.mean(frame)

    if brightness > 170:
        frame = adjust_gamma(frame, 2.0)
    elif brightness < 70:
        frame = adjust_gamma(frame, 0.7)

    return frame, brightness

# ================= GREEN DETECTION =================
def detect_green(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([25,30,30])
    upper = np.array([95,255,255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)

print("SMART PESTICIDE SPRAYER ")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame, brightness = adaptive_vision(frame)
    mask = detect_green(frame)

    h, w = mask.shape

    # 🔥 FIXED VARIABLE NAMES
    left_region = mask[:, :w//3]
    center_region = mask[:, w//3:2*w//3]
    right_region = mask[:, 2*w//3:]

    l = cv2.countNonZero(left_region)
    c = cv2.countNonZero(center_region)
    r = cv2.countNonZero(right_region)

    total = l + c + r
    green_percent = (total / mask.size) * 100

    # ================= DISTANCE =================
    dist = get_distance()

    # ================= SPRAYER (INDEPENDENT) =================
    if SPRAY_ENABLED and c > 1500:
        relay_on()
        spray_status = "SPRAYING"
    else:
        relay_off()
        spray_status = "OFF"

    # ================= INPUT =================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('m'):
        AUTO_MODE = not AUTO_MODE

    if key == ord('p'):
        SPRAY_ENABLED = not SPRAY_ENABLED

    # ================= MOVEMENT =================
    if AUTO_MODE:

        if dist and dist < 20:
            stop()

        elif c > l and c > r:
            forward()

        elif l > r:
            left_turn()

        elif r > l:
            right_turn()

        else:
            forward()

    else:
        if key == ord('w'): forward()
        elif key == ord('s'): backward()
        elif key == ord('a'): left_turn()
        elif key == ord('d'): right_turn()
        elif key == ord('x'): stop()

    # ================= SPEED =================
    if key == ord('+') or key == ord('='):
        speed = min(100, speed + 5)
    elif key == ord('-'):
        speed = max(20, speed - 5)

    # ================= DISPLAY =================
    cv2.putText(frame, f"Green%:{green_percent:.2f}", (10,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(frame, f"L:{l} C:{c} R:{r}", (10,45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.putText(frame, f"Brightness:{int(brightness)}", (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    if dist:
        cv2.putText(frame, f"Dist:{int(dist)}cm", (10,95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.putText(frame, spray_status, (10,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)

    if key == ord('q'):
        break

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
stop()
relay_off()
GPIO.cleanup()
