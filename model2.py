import keras
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from keras import models, optimizers, backend
from keras.layers import core, convolutional, pooling, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def preprocess(img):
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1.4)
    dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_AREA)
    crop_img = dst[109:-1, :]
    x = cv2.resize(crop_img, (32, 16))
    return x


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def preprocess_image_file_train(line_data):
    image = cv2.imread(line_data)

    # finding angle
    jpg_index = line_data.find('.jpg')
    first_index = line_data.find('--') + 3
    last_index = line_data.find('--', first_index) + 2
    y_steer = float(line_data[first_index:last_index - 2])

    image = augment_brightness_camera_images(image)
    image = preprocess(image)
    image = np.array(image)
    ind_flip = np.random.randint(2)
    if ind_flip == 0:
        image = cv2.flip(image, 1)
        y_steer = -y_steer

    return image, y_steer


def generate_train_from_PD_batch(data, batch_size=32):
    batch_images = np.zeros((batch_size, 16, 32, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        a = 0
        while a < batch_size - 1:
            i_line = np.random.randint(len(data))
            line_data = data[i_line]

            x, y = preprocess_image_file_train(line_data)

            if (not (y < -0.1 or y > 0.1)):
                if np.random.random_sample() < 0.6:
                    continue
                else:
                    a = a + 1
            else:
                a = a + 1
            batch_images[a] = x
            batch_steering[a] = y
        # for i_batch in range(batch_size):
        #             i_line = np.random.randint(len(data))
        #             line_data = data[i_line]
        #             keep_pr = 0
        #             #x,y = preprocess_image_file_train(line_data)
        #             x,y = preprocess_image_file_train(line_data)
        #             #x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        #             #y = np.array([[y]])
        #             if(not (y<-0.1 or y > 0.1):
        #                 if(np.random.randint(1)==1):
        #                    i_batch
        #             batch_images[i_batch] = x
        #             batch_steering[i_batch] = y
        #             figure()
        #             plt.imshow(x)
        #             plt.imshow()
        yield batch_images, batch_steering




model = models.Sequential()
model.add(convolutional.Convolution2D(8, 3, 3, input_shape=(16, 32, 3), activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(8, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(core.Dropout(.5))
model.add(core.Flatten())
model.add(core.Dense(50, activation='relu'))
model.add(core.Dense(1))
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

finalset = []
new_size_row = 32
new_size_col = 16

for x in glob.glob('LaptopController/GroundFloor?/*'):
    jpg_index = x.find('.jpg')
    first_index = x.find('--')+3
    last_index = x.find('--',first_index) + 2
    speed = x[last_index:jpg_index]
    if speed == 's1':
        angle = x[first_index:last_index-2]
        finalset.append(x)


model.fit_generator(generate_train_from_PD_batch(finalset,64),samples_per_epoch=1200, verbose=1)

model_json = './model_small.json'
model_h5 = './model_small.h5'
model.save(model_json, model_h5)
