import time

import RPi.GPIO as GPIO

PIN = 14
FREQUENCY = 100  # Hz

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN, GPIO.OUT)

pwm = GPIO.PWM(PIN, FREQUENCY)
pwm.start(0)

try:
    for duty in range(10, 101, 10):
        pwm.ChangeDutyCycle(duty)
        print(f"Set duty cycle to {duty}%")
        time.sleep(2)
finally:
    pwm.stop()
    GPIO.cleanup()