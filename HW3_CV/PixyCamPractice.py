from ev3dev.ev3 import *

lcd = Screen()

# Connect Pixy camera
pixy = Sensor(address=INPUT_3)
assert pixy.connected, "Connecting PixyCam"
# assert pixy.connected, "Error while connecting Pixy camera to port3"

# Connect TouchSensor
ts = TouchSensor(address=INPUT_4)
assert ts.connected, "Connecting TouchSensor"

# Set mode
pixy.mode = 'SIG1'

# # When the mode is set to ALL, you can retrieve data as follows:
# sig = pixy.value(1)*256 + pixy.value(0) # Signature of largest object
# x_centroid = pixy.value(2)    # X-centroid of largest SIG1-object
# y_centroid = pixy.value(3)    # Y-centroid of largest SIG1-object
# width = pixy.value(4)         # Width of the largest SIG1-object
# height = pixy.value(5)        # Height of the largest SIG1-object
#
# # When mode is set to one of the signatures (e.g. SIG1), retrieve data as follows:
# count = pixy.value(0)  # The number of objects that match signature 1
# x = pixy.value(1)      # X-centroid of the largest SIG1-object
# y = pixy.value(2)      # Y-centroid of the largest SIG1-object
# w = pixy.value(3)      # Width of the largest SIG1-object
# h = pixy.value(4)      # Height of the largest SIG1-object

while not ts.value():
  lcd.clear()
  if pixy.value(0) != 0:  # Object with SIG1 detected
    x = pixy.value(1)
    y = pixy.value(2)
    w = pixy.value(3)
    h = pixy.value(4)
    dx = int(w/2)       # Half of the width of the rectangle
    dy = int(h/2)       # Half of the height of the rectangle
    xb = x + int(w/2)   # X-coordinate of bottom-right corner
    yb = y - int(h/2)   # Y-coordinate of the bottom-right corner
    lcd.draw.rectangle((xa,ya,xb,yb), fill='black')
    lcd.update()
