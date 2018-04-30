#!/usr/bin/env python3

import ev3dev.ev3 as ev3


class SturdyBot(object):
    """This provides a higher-level interface to the sturdy Lego robot we've been working
    with."""

    # ---------------------------------------------------------------------------
    # Constants for the configDict
    LEFT_MOTOR = 'left-motor'
    RIGHT_MOTOR = 'right-motor'
    SERVO_MOTOR = 'servo-motor'
    LEFT_TOUCH = 'left-touch'
    RIGHT_TOUCH = 'right-touch'
    ULTRA_SENSOR = 'ultra-sensor'
    COLOR_SENSOR = 'color-sensor'
    GYRO_SENSOR = 'gyro-sensor'
    PIXY = 'pixy-camera'

    # ---------------------------------------------------------------------------
    # Setup methods, including constructor

    def __init__(self, robotName, configDict=None):
        """Takes in a string, the name of the robot, and an optional dictionary
        giving motor and sensor ports for the robot."""
        self.name = robotName
        self.leftMotor = None
        self.rightMotor = None
        self.servoMotor = None
        self.leftTouch = None
        self.rightTouch = None
        self.ultraSensor = None
        self.colorSensor = None
        self.gyroSensor = None
        self.pixy = None

        if configDict is not None:
            self.setupSensorsMotors(configDict)
        if self.leftMotor is None:
            self.leftMotor = ev3.LargeMotor('outC')
        if self.rightMotor is None:
            self.rightMotor = ev3.LargeMotor('outB')

    def setupSensorsMotors(self, configs):
        """Takes in a dictionary where the key is a string that identifies a motor or sensor
        and the value is the port for that motor or sensor. It sets up all the specified motors
        and sensors accordingly."""
        for item in configs:
            port = configs[item]
            if item == self.LEFT_MOTOR:  # "outC"
                self.leftMotor = ev3.LargeMotor(port)
            elif item == self.RIGHT_MOTOR:  # "outB"
                self.rightMotor = ev3.LargeMotor(port)
            elif item == self.SERVO_MOTOR:  # "outD"
                self.servoMotor = ev3.MediumMotor(port)
            elif item == self.LEFT_TOUCH:  # "in4"
                self.leftTouch = ev3.TouchSensor(port)
            elif item == self.RIGHT_TOUCH:  # "in1"
                self.rightTouch = ev3.TouchSensor(port)
            elif item == self.ULTRA_SENSOR:  # "in3"
                self.ultraSensor = ev3.UltrasonicSensor(port)
            elif item == self.COLOR_SENSOR:  # "in2"
                self.colorSensor = ev3.ColorSensor(port)
            elif item == self.GYRO_SENSOR:  # not connected
                self.gyroSensor = ev3.GyroSensor(port)
            elif item == self.PIXY:
                self.pixy = ev3.Sensor(port)
            else:
                print("Unknown configuration item:", item)

    def setMotorPort(self, side, port):
        """Takes in which side and which port, and changes the correct variable
        to connect to that port."""
        if side == self.LEFT_MOTOR:
            self.leftMotor = ev3.LargeMotor(port)
        elif side == self.RIGHT_MOTOR:
            self.rightMotor = ev3.LargeMotor(port)
        elif side == self.SERVO_MOTOR:
            self.servoMotor = ev3.MediumMotor(port)
        else:
            print("Incorrect motor description:", side)

    def setTouchSensor(self, side, port):
        """Takes in which side and which port, and changes the correct
        variable to connect to that port"""
        if side == self.LEFT_TOUCH:
            self.leftTouch = ev3.TouchSensor(port)
        elif side == self.RIGHT_TOUCH:
            self.rightTouch = ev3.TouchSensor(port)
        else:
            print("Incorrect touch sensor description:", side)

    def setColorSensor(self, port):
        """Takes in the port for the color sensor and updates object"""
        self.colorSensor = ev3.ColorSensor(port)

    def setUltrasonicSensor(self, port):
        """Takes in the port for the ultrasonic sensor and updates object"""
        self.ultraSensor = ev3.UltrasonicSensor(port)

    def setGyroSensor(self, port):
        """Takes in the port for the gyro sensor and updates object"""
        self.gyroSensor = ev3.GyroSensor(port)

    def setPixy(self, port):
        self.pixy = ev3.Sensor(port)
        self.pixy.mode = 'SIG1'

    # ---------------------------------------------------------------------------
    # Methods to read sensor values
    def readTouch(self):
        """Reports the value of both touch sensors, OR just one if only one is connected, OR
        prints an alert and returns nothing if neither is connected."""
        if self.leftTouch is not None and self.rightTouch is not None:
            return self.leftTouch.is_pressed, self.rightTouch.is_pressed
        elif self.leftTouch is not None:
            return self.leftTouch.is_pressed, None
        elif self.rightTouch is not None:
            return None, self.rightTouch
        else:
            print("Warning, no touch sensor connected")
            return None, None

    def readReflect(self):
        """Reports the reflectance value for the color sensor, OR
        prints an alert and returns nothing if the sensor is not connected."""
        if self.colorSensor is not None:
            return self.colorSensor.reflected_light_intensity
        else:
            print("Warning, no color sensor connected")
            return None

    def readAmbient(self):
        """Reports the reflectance value for the color sensor, OR
        prints an alert and returns nothing if the sensor is not connected."""
        if self.colorSensor is not None:
            return self.colorSensor.ambient_light_intensity
        else:
            print("Warning, no color sensor connected")
            return None

    def readColor(self):
        """Reports the color value (0 through 7) the color sensor, OR
        prints an alert and returns nothing if the sensor is not connected."""
        if self.colorSensor is not None:
            return self.colorSensor.color
        else:
            print("Warning, no color sensor connected")
            return None

    def readDistance(self):
        """Report the ultrasonic sensor’s value in centimeters, OR
        prints an alert and returns nothing if the sensor is not connected."""
        if self.ultraSensor is not None:
            return self.ultraSensor.distance_centimeters
        else:
            print("Warning, no ultrasonic sensor connected")
            return None

    def readHeading(self):
        """Report the gyro sensor’s value, adjusting it to be between 0 and 360, OR
        prints an alert and returns nothing if the sensor is not connected."""
        if self.gyroSensor is not None:
            return self.gyroSensor.angle
        else:
            print("Warning, no ultrasonic sensor connected")
            return None

    # ---------------------------------------------------------------------------
    # Methods to move robot

    # Put your code here, make changes to make it consistent

    def forward(self, speed, time=None):
        """This method takes in a speed, which is given as a real number between -1.0 and +1.0. It
        also has an optional second input which is the amount of time, in seconds, that we want
        the robot to run. The method should set the left and right motors to a speed, scaling it
        so that 1.0 maps to the maximum speed of the motor, and should cause the robot to move
        straight forward, either turning on the motors indefinitely or for the input time.
        Negative speeds should make the robot move straight backwards.
        """
        # Range check
        if speed > 1.0 or speed < -1.0:
            exit('Speed of range [-1.0, 1.0] needed')
        if time is None:
            self.leftMotor.run_forever(speed_sp=self.leftMotor.max_speed * speed)
            self.rightMotor.run_forever(speed_sp=self.rightMotor.max_speed * speed)
        else:
            self.leftMotor.run_timed(speed_sp=self.leftMotor.max_speed * speed, time_sp=time * 1000)
            self.rightMotor.run_timed(speed_sp=self.rightMotor.max_speed * speed, time_sp=time * 1000)
            self.leftMotor.wait_until_not_moving()

    def backward(self, speed, time=None):
        """This method is similar to the previous, but positive speeds move the robot backwards and
        negative move it forwards
        """
        if speed > 1.0 or speed < -1.0:
            exit('Speed of range [-1.0, 1.0] needed')  # raise an exception instead?
        if time is None:
            self.leftMotor.run_forever(speed_sp=self.leftMotor.max_speed * -1 * speed)
            self.rightMotor.run_forever(speed_sp=self.rightMotor.max_speed * -1 * speed)
        else:
            self.leftMotor.run_timed(speed_sp=self.leftMotor.max_speed * -1 * speed, time_sp=time * 1000)
            self.rightMotor.run_timed(speed_sp=self.rightMotor.max_speed * -1 * speed, time_sp=time * 1000)
            self.leftMotor.wait_until_not_moving()

    def turnRight(self, speed, time=None):
        """This function should cause the robot to rotate counter-clockwise at the given speed either
        indefinitely or until a given time. Negative speeds should cause it to rotate clockwise.
        """
        if speed > 1.0 or speed < -1.0:
            exit('Speed of range [-1.0, 1.0] needed')  # raise an exception instead?
        if time is None:
            self.leftMotor.run_forever(speed_sp=self.leftMotor.max_speed * speed)
        else:
            self.leftMotor.run_timed(speed_sp=self.leftMotor.max_speed * speed, time_sp=time * 1000)
            self.leftMotor.wait_until_not_moving()

    def turnLeft(self, speed, time=None):
        """This function should cause the robot to rotate clockwise at the given speed either
        indefinitely or until a given time. Negative speeds should cause it to rotate
        counter-clockwise.
        """
        if speed > 1.0 or speed < -1.0:
            exit('Speed of range [-1.0, 1.0] needed')  # raise an exception instead?
        if time is None:
            self.rightMotor.run_forever(speed_sp=self.rightMotor.max_speed * speed)
        else:
            self.rightMotor.run_timed(speed_sp=self.rightMotor.max_speed * speed, time_sp=time * 1000)
            self.rightMotor.wait_until_not_moving()

    def stop(self):
        """This function should cause the robot to stop, turning off the left and right motors."""
        self.leftMotor.stop()
        self.rightMotor.stop()

    def curve(self, leftSpeed, rightSpeed, time=None):
        """This function should cause the robot to move in a curve by having different speeds
        for left and right motors. This is a bit more low-level than the previous functions,
        but the speeds should still be in the range from -1.0 to +1.0.
        """
        if leftSpeed > 1.0 or leftSpeed < -1.0 or rightSpeed > 1.0 or rightSpeed < -1.0:
            exit('Speed of range [-1.0, 1.0] is needed for both right and left wheels')
        if time is None:
            self.leftMotor.run_forever(speed_sp=leftSpeed * self.leftMotor.max_speed)
            self.rightMotor.run_forever(speed_sp=rightSpeed * self.rightMotor.max_speed)
        else:
            self.leftMotor.run_timed(speed_sp=leftSpeed * self.leftMotor.max_speed, time_sp=time * 1000)
            self.rightMotor.run_timed(speed_sp=rightSpeed * self.rightMotor.max_speed, time_sp=time * 1000)
            self.leftMotor.wait_until_not_moving()

    def honeybee(self):
        """Finds and approaches a brightly colored object. It will live in a
        maze of some kind, at least it should robustly deal with obstacles and
        narrow passageways. The robot should wander around, looking for a
        brightly colored object.
        """
        # The song for finding the brightest color.
        ev3.Sound.set_volume(10)
        starSong_long = [
            ('C4', 'q'), ('C4', 'q'), ('G4', 'q'), ('G4', 'q'),
            ('A4', 'q'), ('A4', 'q'), ('G4', 'h'),
            ('F4', 'q'), ('F4', 'q'), ('E4', 'q'), ('E4', 'q'),
            ('D4', 'q'), ('D4', 'q'), ('C4', 'h')]

        # The robot starts out turning right. This direction change is for the
        # sake of randomness.
        turnRightWhenBumping = True

        # Quit the program if any button is pressed.
        while (not buttons.any()):
            # Check and update the sensor values to use later.
            touchValues = self.readTouch()
            distValues = self.readDistance()

            # Check if found the brightest color value, which is set manually as
            # a parameter. If found, stop the robot and play the triumphant song.
            # Then quit the program.
            if self.pixy.value(0) != 0:
                while True:
                    self.forward(0.05, 0.8)
                    print(self.pixy.value(2))
                    print(self.pixy.value(3))
                    print(self.pixy.value(4))
                    if (self.pixy.value(2) > 80 and self.pixy.value(2) < 180) and (
                            self.pixy.value(3) > 80 and self.pixy.value(3) < 120) and (
                            self.pixy.value(4) > 130 and self.pixy.value(4) < 180):
                        ev3.Sound.play_song(starSong_long)
                        while True:
                            self.stop()
                        break

            # If the wall the robot is facing is more than 30cm away,change the
            # turning direction.
            if distValues > 30:
                turnRightWhenBumping = False
            if turnRightWhenBumping:  # turning right after bumping into the wall
                if touchValues == (True, True):  # both arms touched, move backwards then turn right
                    self.backward(0.1, 2)
                    self.turnRight(0.1, 1.5)
                elif touchValues == (False, False):  # not close enough to the wall
                    self.forward(0.05, 0.8)
                elif touchValues == (True, False):  # left arm touched
                    self.backward(0.1, 0.5)
                    self.turnLeft(0.05, 0.4)
                elif touchValues == (False, True):  # right arm touched
                    self.backward(0.1, 0.5)
                    self.turnRight(0.05, 0.4)
            else:  # turning left after bumping into the wall
                if touchValues == (True, True):  # both arms touched, move backwards
                    self.backward(0.1, 2)
                    self.turnLeft(0.1, 1.5)
                elif touchValues == (False, False):  # not close enough to the wall
                    self.forward(0.05, 0.8)
                elif touchValues == (True, False):  # left arm touched
                    self.backward(0.1, 0.5)
                    self.turnLeft(0.05, 0.4)
                elif touchValues == (False, True):  # right arm touched
                    self.backward(0.1, 0.5)
                    self.turnRight(0.05, 0.4)

        self.stop()


if __name__ == "__main__":
    buttons = ev3.Button()
    ev3.Sound.set_volume(100)
    ev3.Sound.speak("Starting")

    honeybeeConfig = {SturdyBot.LEFT_MOTOR: 'outC',
                      SturdyBot.RIGHT_MOTOR: 'outB',
                      SturdyBot.LEFT_TOUCH: 'in4',
                      SturdyBot.RIGHT_TOUCH: 'in1',
                      SturdyBot.COLOR_SENSOR: 'in2',
                      SturdyBot.ULTRA_SENSOR: 'in3',
                      SturdyBot.PIXY: 'in2:i2c1'}
    honeybeeRobot = SturdyBot('Honeybee', honeybeeConfig)
    honeybeeRobot.honeybee()

    ev3.Sound.speak("Done")
