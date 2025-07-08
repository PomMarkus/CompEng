import time

import RPi.GPIO as GPIO

PIN = 15

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN, GPIO.OUT)

try:
    for pulse_time in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print(f"\nPulse time: {pulse_time} s")
        for i in range(3):
            GPIO.output(PIN, GPIO.HIGH)
            # print(f"  Pulse {i+1}: Motor ON for {pulse_time} s")
            time.sleep(pulse_time)
            GPIO.output(PIN, GPIO.LOW)
            print(f"  Pulse {i+1}: Motor OFF")
            time.sleep(pulse_time)  # Pause between pulses
        time.sleep(1)  # Pause between different pulse_time sets
finally:
    GPIO.output(PIN, GPIO.LOW)
    GPIO.cleanup()