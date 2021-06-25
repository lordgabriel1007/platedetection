from bluedot.btcomm import BluetoothServer
from signal import pause
import RPi.GPIO as GPIO
import time


#Set function to calculate percent from angle
def angle_to_percent (angle) :
    if angle > 180 or angle < 0 :
        return False

    start = 4
    end = 12.5
    ratio = (end - start)/180 #Calcul ratio from angle to percent

    angle_as_percent = angle * ratio

    return start + angle_as_percent

#INITIALIZE GPIO
GPIO.setmode(GPIO.BOARD) #Use Board numerotation mode
GPIO.setwarnings(False) #Disable warnings

#Use pin 12 for PWM signal
pwm_gpio = 12
frequence = 50
GPIO.setup(pwm_gpio, GPIO.OUT)
pwm = GPIO.PWM(pwm_gpio, frequence)

#Init at 0°
pwm.start(angle_to_percent(0))
time.sleep(1)




def data_received(data):
    print(data)
    if data=='o':
        pwm.start(angle_to_percent(150))
    elif data=='c':
        pwm.start(angle_to_percent(0))
    elif data=='x':
        pwm.stop()
        GPIO.cleanup()
        print('servo stop')
        exit()
    
s = BluetoothServer(data_received, port=5)

pause()
