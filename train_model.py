import csv
import cv2
import numpy as np
import os
import glob
import pre_processing

path = './data'
extension = 'csv'
existing_csvs = [i for i in glob.glob('data/*.{}'.format(extension))]
print('Training with %s files' % len(existing_csvs))

# Put all training data from multiple files in the folder into the lines array
lines = []
for csv_file_name in existing_csvs:

    with open('./'+csv_file_name) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

print('%s data points' % len(lines))

from sklearn.model_selection import train_test_split
import sklearn
from random import shuffle

train_samples, validation_samples = train_test_split(lines, test_size=0.3)
batch_size = 64
EPOCHS = 5
WIDTH, HEIGHT = pre_processing.size[0], pre_processing.size[1]

def generator(samples, batch_size=32):

    samples = np.array(samples)
    num_samples = len(samples)
    while 1:
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            # Start of the batch pre-processing
            # print("batch {}".format(offset))

            batch_samples = samples[offset:offset + batch_size]
            images = []
            measurements = []

            for batch_sample in batch_samples:

                side_camera_correction = 0.2
                for i in range(3):

                    side_image_threshold = 0.3
                    measurement = float(batch_sample[3])
                    # Skip augmentation if it's a straight angle
                    if i != 0 and (-side_image_threshold > measurement > side_image_threshold):
                        continue

                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = "./data/IMG/" + filename
                    image = cv2.imread(current_path)

                    # image = image[55:135, 0:320]
                    image = pre_processing.process(image)
                    image = cv2.resize(image, (WIDTH, HEIGHT))
                    images.append(image)

                    # if measurement < side_image_threshold and measurement > -side_image_threshold:
                    if i == 1:
                        measurement += side_camera_correction
                    if i == 2:
                        measurement -= side_camera_correction
                    measurement = min(max(measurement, -1.0), 1.0)
                    measurements.append(float(measurement))

            # Augment the images by flipping the image and the measurement
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)

                # Flipped
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)

            yield sklearn.utils.shuffle(X_train, y_train)

augmentation_factor = 2
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Input, Layer
from keras.layers import Cropping2D, Dropout, Reshape

def custom_model():

    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(HEIGHT, WIDTH, 3)))
    dropout = 0.2
    model.add(Dropout(dropout))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Flatten())
    # model.add(Dense(1164))
    # model.add(Dropout(dropout))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.fit_generator(train_generator, samples_per_epoch=(len(train_samples) * augmentation_factor) / 1,
                        validation_data=validation_generator,
                        nb_val_samples=(len(validation_samples) * augmentation_factor)/2, nb_epoch=EPOCHS)

    model.save('model.h5')
custom_model()

def vgg_network():
    from keras.applications.vgg16 import VGG16
    from keras.callbacks import ModelCheckpoint
    from keras.models import Model
    from keras.layers import Dense

    input_shape = (HEIGHT, WIDTH, 3)
    input_tensor = Input(shape=input_shape)
    color_norm = Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape)(input_tensor)
    crop = Cropping2D(cropping=((15, 3), (0, 0)))(color_norm)

    base_model = VGG16(input_tensor=crop, weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    # Add top layer for regression
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024)(x)
    # x = Dropout(0.2)(x)
    x = Dense(100)(x)
    # x = Dropout(0.2)(x)
    x = Dense(50)(x)
    x = Dense(10)(x)
    regression = Dense(1)(x)

    model = Model(base_model.input, regression)

    checkpoint = ModelCheckpoint('model-vgg-cp.h5', monitor='val_acc', verbose=1, save_best_only=False, mode='max')

    model.compile(loss='mae', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * augmentation_factor,
                        validation_data=validation_generator,
                        callbacks=[checkpoint],
                        nb_val_samples=len(validation_samples) * augmentation_factor, nb_epoch=EPOCHS)

    model.save('model-vgg.h5')
# vgg_network()





def inceptionv3_network():
    from keras.applications.inception_v3 import InceptionV3
    from keras.callbacks import ModelCheckpoint
    from keras.models import Model
    from keras.layers import Dense

    input_tensor = Input(shape=(160, 320, 3))
    color_norm = Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3))(input_tensor)
    crop = Cropping2D(cropping=((70, 10), (0, 0)))(color_norm)

    base_model = InceptionV3(input_tensor=crop, input_shape=(160, 320, 3), weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    # Add top layer for regression
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Dense(120)(x)
    regression = Dense(1)(x)

    model = Model(base_model.input, regression)

    checkpoint = ModelCheckpoint('model-inception-cp.h5', monitor='val_acc', verbose=1, save_best_only=False, mode='max')

    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                        validation_data=validation_generator, callbacks=[checkpoint],
                        nb_val_samples=len(validation_samples), nb_epoch=3)

    model.save('model-inception.h5')
# inceptionv3_network()
