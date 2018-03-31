import os
import csv
import cv2
import numpy as np

# Allows me to train on one GPU and evaluate on another
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Where my data is
data_dir = "data"

# Get lines from the csv
def get_lines(filename, data_dir):
    lines = []
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # Create raw paths i.e. IMG/file
            center = "/".join(line[0].strip().split("/")[-2:])
            left = "/".join(line[1].strip().split("/")[-2:])
            right = "/".join(line[2].strip().split("/")[-2:])

            center_path = os.path.join(data_dir, center)
            left_path = os.path.join(data_dir, left)
            right_path = os.path.join(data_dir, right)

            # Use the side camera to create extra data
            correction_factor = 0.25
            steering_angle = float(line[3])
            steering_left = steering_angle + correction_factor
            steering_right = steering_angle - correction_factor

            if cv2.imread(center_path) is None or cv2.imread(left_path) is None or cv2.imread(right_path) is None:
                continue

            lines.append([center_path, steering_angle])            
            lines.append([left_path, steering_left])
            lines.append([right_path, steering_right])
    return lines

# Get the data from csv
lines = get_lines("./data/driving_log.csv", data_dir)
lines2 = get_lines("./data/driving_log1.csv", data_dir)
lines.extend(lines2)
lines2 = get_lines("./data/driving_log2.csv", data_dir)
lines.extend(lines2)
print(len(lines))

# Split the data into train and test
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
X_train, X_valid = train_test_split(lines, test_size=0.2)

print(len(X_train))
print(len(X_valid))

batch_size = 128

# process the data BGR -> YUV because it used cv2 to read images
def preprocess_image(img):
    new_img = img[50:140, :, :]
    new_img = cv2.resize(new_img, (200, 66), interpolation=cv2.INTER_AREA)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img

# Generator to read in and process images
def generator(lines, batch_size=32):
    num_samples = len(lines)
    while 1:
        shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_lines = lines[offset:offset+batch_size]
            images = []
            angles = []
            for line in batch_lines:
                
                image = cv2.imread(line[0])
                image = preprocess_image(image)
                steering_angle = line[1]

                images.append(image)
                angles.append(steering_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_gen = generator(X_train, batch_size=batch_size)
valid_gen = generator(X_valid, batch_size=batch_size)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D, Input, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

# Build my model based on the NVIDIA architecture
model = Sequential()
model.add(Lambda(lambda x: x / 127. - 1., input_shape=(66, 200, 3)))
model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.summary()

# Compile loss
model.compile(loss='mse', optimizer=Adam(lr=1e-4))
# Callback to only save the best val loss
filepath = "./models/model-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# Fit the the model using the generator.
model.fit_generator(train_gen, steps_per_epoch=len(X_train)//batch_size, validation_data=valid_gen, validation_steps=len(X_valid)//batch_size, callbacks=callbacks_list, epochs=15)

model.save('model.h5')
