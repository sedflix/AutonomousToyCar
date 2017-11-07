import pygame
import pygame.camera
from pygame.locals import *
from datetime import datetime

from BluetoothCom import BluetoothComm

DEVICE = '/dev/video0'
SIZE = (640, 480)
FOLDER = 'img/'


serverMACAddress = '00:15:83:35:99:09'

bluetoothServer = BluetoothComm(serverMACAddress, True)


def camstream():
    # initialising pygame display
    pygame.init()
    display = pygame.display.set_mode(SIZE, 0)

    # initialising camera
    pygame.camera.init()
    camera = pygame.camera.Camera(DEVICE, SIZE)
    camera.start()

    # initialising joystick
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
    print(joysticks)

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    screen = pygame.surface.Surface(SIZE, 0, display)
    capture = True
    recording = False
    x = 0
    y = 0
    while capture:
        # screen = camera.get_image(screen)
        # display.blit(screen, (0, 0))
        pygame.display.flip()
        for event in pygame.event.get():

            lf = -joystick.get_axis(0)
            leftright = int(translate(lf,-1,1,20,160))
            updown = -(round(joystick.get_axis(1), 3))

            angleInfo = 'a' + formatAngle(leftright)
            speedInfo = 's' + formatSpeed(updown)

            # saveImage(screen, lf, recording)
            bluetoothServer.send(angleInfo)
            if (not updown == 0):
                bluetoothServer.send(speedInfo)



            if event.type == QUIT:
                capture = False

            # i:0 -> 1
            # i:1 -> 2
            # i:2 -> 3
            # i:3 -> 4
            # i:4 -> L1
            # i:5 -> R1
            # i:6 -> L2
            # i:7 -> R2
            if event.type == pygame.JOYBUTTONDOWN:

                if joystick.get_button(5) == 1:
                    # R1
                    recording = True
                    print("recording status : " + str(recording))

                if joystick.get_button(7) == 1:
                    # R2
                    recording = False
                    print("recording status : " + str(recording))

                if joystick.get_button(4) == 1 or joystick.get_button(6) == 1:
                    # L1 or L2
                    recording = False
                    bluetoothServer.send("s3")
                    print ("Stoping")
                    print("recording status : " + str(recording))

            clock.tick(60)

    camera.stop()
    pygame.quit()
    return


def saveImage(img, event_angle, recording):
    if (recording):
        pygame.image.save(img, FOLDER + str(datetime.now()) + "--" + str(event_angle) + ".jpg")


def formatAngle(x):
    # x = x*90
    # if x >= 0:
    #     x = x + 90
    # else:
    #     x = x + 90
    # x = round(x,0)
    # x = int(x)
    x = str(x)
    if len(x) == 1:
        return '00' + x
    if len(x) == 2:
        return '0' + x
    return x


def formatSpeed(x):
    if x < 0:
        x = 2  # back
    elif x > 0:
        x = 1  # forward
    else:
        x = 3  # stop

    return str(x)


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


if __name__ == '__main__':
    camstream()
