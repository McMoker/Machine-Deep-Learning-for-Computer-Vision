## Using ImageDataGenerator() to Classify Galaxies using image data provided by Galaxy Zoo via Codecademy
## Currently, code works solely for codecademy terminal output where project was assembled

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
## Codecademy Utilities
from utils import load_galaxy_data
## Codecademy app
import app

## Custom function load_galaxy_data() loads the compressed data files into the Codecademy learning environment as NumPy arrays:
input_data, labels = load_galaxy_data()
## Use .shape to print the dimensions of the input_data and labels.
# print(input_data.shape)
# print(labels.shape)
## Successfully loaded galaxy data!
## (1400, 128, 128, 3)
## (1400, 4)

## Divide the data into training and validation data using sklearn's train_test_split() function:
x_train, x_valid, y_train, y_valid = train_test_split(input_data, labels, test_size=0.20, stratify=labels, shuffle=True, random_state=222)

## ImageDataGenerator to process input, rescale to normalize pixels
data_generator = ImageDataGenerator(rescale=1.0/255)

## Create two NumpyArrayIterators using .flow() method. batch_size=5
training_iterator = data_generator.flow(x_train, y_train, batch_size=5)
validation_iterator = data_generator.flow(x_train, y_train, batch_size=5)

## Build model using keras.Sequential()
model = tf.keras.Sequential()
## Input Layer
model.add(tf.keras.Input(shape=(128, 128, 3)))
## Convolutional Layer 1 - 8 filters, 3x3 with strides of 2
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation='relu'))
## Max Pooling 2D
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
## Convolutional Layer 2 - 8 filters, 3x3 with strides of 2
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation='relu'))
# Max Pooling 2D
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Flatten Layer
model.add(tf.keras.layers.Flatten())
# Hidden Dense Layer with 16 hidden units
model.add(tf.keras.layers.Dense(16, activation='relu'))
## Output Layer
model.add(tf.keras.layers.Dense(4, activation='softmax'))
## Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()])

# model.summary()
## 7,164 Total parameters
model.fit(training_iterator, steps_per_epoch=len(x_train)/5, epochs=12, validation_data=validation_iterator, validation_steps=len(x_valid)/5)

## Model complete // 