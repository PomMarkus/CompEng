#!/usr/bin/python3
import threading
from time import sleep
from datetime import datetime as dt
import RPi.GPIO as GPIO
#import rpi.GPIO as GPIO

PIN = 15

GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN, GPIO.OUT)

def high(pin):
    GPIO.output(pin,GPIO.HIGH)
def low(pin):
    GPIO.output(pin,GPIO.LOW)

output=False
stop=False
time=0


def control():
    global output
    global stop
    
    print("Enter \"0\" to stop the procedure\nEnter \"1\" to toggle the output")
    
    while(not stop):
        try:
            var=int(input())
        except:
            print("invalid input")
            continue
        
        if var==0:
            stop=True
        elif var==1 and output==False:
            output=True
        elif var==1 and output==True:
            output=False
        else:
            print("invalid input")
            
    
def tempcheck():
    global output
    global stop
    global time
    tempData = "/sys/class/thermal/thermal_zone0/temp"

    def reader():
        read = open(tempData, "r")
        x=int(read.readline(2))
        read.close()
        return(x)

    temps=[]

    for i in range(10):
        temps.append(0)
        
    time0=0
    timeold=0
    times=[]
    for i in range(10):
        times.append(0)
        
    t=[0,58,61,64,67,70,73,76,78,100]
    speed=[0,30,40,50,60,70,80,90,100]

    while not stop:
        for _ in range(10):
            temp=reader()
            temps.append(temp)
            temps.pop(0)
            sleep(1)

        mtemp=sum(temps)//len(temps)
        if output:
            print("Mean temp. is: " + str(mtemp) + "Â°C")

        for i in range(9):
            if  t[i+1] >= mtemp > t[i]:
                time0=speed[i]

        #print(time0)
        times.append(time0)
        times.pop(0)
        time=max(times)
        if time != timeold:
            timeold=time
            print(time, dt.now())
        #print(time)

def pwm(pin=PIN):
    global stop
    global time
    #run=False
    while not stop:
        if time==0:
            #if run==True:
            #    run=False
            low(pin)
            sleep(1)
        elif time==100:
            high(pin)
            sleep(1)
        else:
            #if run==False:
            #    print("startup")
            #    run=True
            delay1=time*1E-5
            delay2=(100-time)*1E-5
            high(pin)
            sleep(delay1)
            low(pin)
            sleep(delay2)
    low(pin)
    
c=threading.Thread(target=control)
t=threading.Thread(target=tempcheck)
p=threading.Thread(target=pwm)

c.start()
t.start()
p.start()
c.join()
t.join()
p.join()

GPIO.cleanup()
print("terminated as expected")
