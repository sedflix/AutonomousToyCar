import pygame
import pygame.camera
from pygame.locals import *
from datetime import datetime

DEVICE = '/dev/video0'
SIZE = (640, 480)
FILENAME = 'capture.png'
FOLDER = 'img/'


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

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    screen = pygame.surface.Surface(SIZE, 0, display)
    capture = True
    recording = False
    x = 0
    y = 0
    while capture:
        screen = camera.get_image(screen)
        display.blit(screen, (0, 0))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == QUIT:
                capture = False
            elif event.type == KEYDOWN and event.key == K_r:
                recording = not recording
                print("recording status : " + str(recording))

            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_UP]:
                y -= 3
                saveImage(screen,"0",recording)
            if pressed[pygame.K_DOWN]:
                y += 3
                saveImage(screen, "0", recording)
            if pressed[pygame.K_LEFT]:
                x -= 3
                saveImage(screen, "-35", recording)
            if pressed[pygame.K_RIGHT]:
                x += 3
                saveImage(screen, "35", recording)


            print(str(x) + ", " + str(y))

            clock.tick(20)

    camera.stop()
    pygame.quit()
    return


def saveImage(img, event_angle, recording):
    if(recording):
        pygame.image.save(img, FOLDER + str(datetime.now()) + "--" + str(event_angle) + ".jpg")


if __name__ == '__main__':
    camstream()
