import time

import RPi.GPIO as GPIO

PIN = 14

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN, GPIO.OUT)

try:
    for pulse_time in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        GPIO.output(PIN, GPIO.HIGH)
        print(f"Motor ON for {pulse_time} s")
        time.sleep(pulse_time)
        GPIO.output(PIN, GPIO.LOW)
        print("Motor OFF")
        time.sleep(1)  # Pause between pulses
finally:
    GPIO.output(PIN, GPIO.LOW)
    GPIO.cleanup()