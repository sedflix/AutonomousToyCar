# AutonomousToyCar
Self-Driving Toy Car using End-To-End Learning Model

## Car and Data Collection

**Actively looking for a proper RC car** 

This project involved making a toy car from scratch in a very restriced budget. We took a cheap RC car from a friend and hacked it. It had discrete steering(eihter 45 deg left or right). So we replaced a motor with a servo motor(It was a pain), to get a proper steering method. Motors were connected to Arduino, which was connected to a Buletooth module. We controlled the car using a game controller, which was connected to the laptop. A phone was attached to the car to act as a camera. 

[ServoDCViaBluetooth.ino](https://github.com/geekSiddharth/AutonomousToyCar/blob/master/Arduino/ServoDCViaBluetooth/ServoDCViaBluetooth.ino) contains the Bluetooth code for the car. 

[BluetoothCom.py](https://github.com/geekSiddharth/AutonomousToyCar/blob/master/LaptopController/BluetoothCom.py) is a [**PyBluez**](https://pypi.python.org/pypi/PyBluez) wrapper for our purpose.


[CarController.py](https://github.com/geekSiddharth/AutonomousToyCar/blob/master/LaptopController/CarController.py) is the main code that runs on the laptop. It takes input from the phone and the game controller. It sends command to the car. And it records the data (Images at a rate of 60fps and with name in the format of `datetime.now() + "--" + event_angle + "--" + upORdown + ".jpg"`)

**Note: The car had a very large turning radius. Sometimes there was considerale lag in the Bluetooth Communication. Therfore controlling the car became really difficult. Therefore the dataset collected is not that nice**

## Models

We did heavy data augumentation. Credits for data augumentation cod goes to this [blog](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9). 
Due to absence of computing power we relied on small CNN networks rather than the larger one. We tried two networks:

#### Smallest one (Inspired by SqueezeNet):

```
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
```

#### The larger network:

``` 
  model = models.Sequential()
  model.add(convolutional.Convolution2D(16, 3, 3, input_shape=(84, 128, 3), activation='relu'))
  model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
  model.add(convolutional.Convolution2D(16, 3, 3, activation='relu'))
  model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
  model.add(convolutional.Convolution2D(32, 3, 3, activation='relu'))
  model.add(pooling.MaxPooling2D(pool_size=(4, 4)))
  model.add(Dropout(0.5))
  model.add(core.Flatten())
  model.add(core.Dense(500, activation='relu'))
  model.add(core.Dropout(.5))
  model.add(core.Dense(100, activation='relu'))
  model.add(core.Dropout(.25))
  model.add(core.Dense(20, activation='relu'))
  model.add(core.Dense(1))
  adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(loss='mean_squared_error', optimizer='adam')
```

**Surprise: the smaller network gave much better performace**


## Plans:

- [ ] Get a better car
- [ ] Get access to a better workstation(so that we can use deeper networks)
- [ ] Implement online learning/live training
- [ ] Fine-tune similar networks(which are trained on real world and real car data) and test it with car
- [ ] Customise it for indoor navigation
- [ ] ...- [ ] Get a better car
67
- [ ] Get access to a better workstation(so that we can use deeper networks)
68
- [ ] Implement online learning/live training
69
- [ ] Customise it for indoor navigation