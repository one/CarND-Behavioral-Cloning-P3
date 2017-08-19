### constant parameters
READ_PICS_LIMIT = 100000  # limit the number of pictures that will be read in

EPOCHS = 5
BATCH_SIZE = 32
VALID_SPLIT = 0.2

### imports
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.image as mpimg
import csv
import cv2
import re
from sklearn.utils import shuffle

### load, shuffle and split data
lines = []
with open ('data/driving_log.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    for line in csv_reader:
        lines.append(line)  # format: center,left,right,steering,throttle,brake,speed

# shuffle the lines before loading the images
lines = shuffle(lines)

images = []
measurements = []
for i, line in enumerate(lines):
    if i >= READ_PICS_LIMIT:
        # break if the specified limit of images is reached
        break
    source_path = line[0]
    source_path = source_path.replace('\\', '/')
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

print("loaded %s images"%len(images))
print("loaded %s steering angles"%len(measurements))

imgs_flipped = []
for img in images:
    imgs_flipped.append(np.fliplr(img))
    pass
measurements_flipped = []
for angle in measurements:
    measurements_flipped.append(-angle)
    pass

X_train = np.array(images + imgs_flipped)
y_train = measurements + measurements_flipped

### define keras network
# Initial Setup for Keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Lambda, BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
#model.add(Lambda(lambda x: x / 255. - .5, input_shape=(160, 320, 3))) --> this doesn't work in my environment, so I use the next line as an alternative
model.add(BatchNormalization(input_shape=(160, 320, 3), axis=1))

model.add(Cropping2D(cropping=((67,25),(0,0))))

model.add(Convolution2D(24, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Flatten())

model.add(Dense(100))
model.add((Dropout(0.5)))
model.add(Activation('relu'))
model.add(Dense(50))
model.add((Dropout(0.5)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add((Dropout(0.5)))
model.add(Activation('relu'))
model.add(Dense(1))
    
### define training operations
model.compile(optimizer='adam', loss='mse')
history_object = model.fit(X_train, y_train, nb_epoch=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALID_SPLIT, shuffle=True)

### save the model
model.save('model.h5')

### plot the training and validation loss for each epoch
import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
