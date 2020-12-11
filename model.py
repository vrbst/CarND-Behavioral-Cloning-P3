import os
import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers.pooling import MaxPooling2D
from math import ceil
from random import shuffle

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    path = './data/IMG/'
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_name)
                center_image[:,:,::-1]
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                left_name = './data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_name)
                left_image[:,:,::-1]
                left_angle = center_angle + 0.2
                images.append(left_image)
                angles.append(left_angle)
                
                right_name = './data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_name)
                right_image[:,:,::-1]
                right_angle = center_angle - 0.2
                images.append(right_image)
                angles.append(right_angle)                

            augmented_images, augmented_measurements = [] , []
            for image, measurement in zip(images, angles):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)
            
            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255.0 - 0.5,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(48, (5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(Flatten())
#model.add(Dropout(0.25))
model.add(Dense(100))
#model.add(Dropout(0.25))
model.add(Dense(50))
#model.add(Dropout(0.25))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size) * 6, validation_data=validation_generator, validation_steps=ceil(len(validation_samples)/batch_size), epochs=3, verbose=1)

model.save('model.h5')