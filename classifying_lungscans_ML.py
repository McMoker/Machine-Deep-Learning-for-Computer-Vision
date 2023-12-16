import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy

## Variables for Data Generator
DIRECTORY = "lung_scans_Covid19_project\lungscans_Covid19_dataset\train"
CLASS_MODE = "categorical"
COLOR_MODE = "grayscale"
TARGET_SIZE = (256, 256)
BATCH_SIZE = 30

## Assembling Data Generators
training_data_generator = ImageDataGenerator(rescale=1.0/255, zoom_range=0.15, rotation_range=20, width_shift_range=0.05, height_shift_range=0.05)

validation_data_generator = ImageDataGenerator()
training_iterator = training_data_generator.flow_from_directory(DIRECTORY, class_mode='categorical', color_mode='grayscale', batch_size=BATCH_SIZE)

## Assembling Iterators
training_iterator.next()
print("\nLoading validation data...")

validation_iterator = validation_data_generator.flow_from_directory(DIRECTORY, class_mode='categorical', color_mode='grayscale', batch_size=BATCH_SIZE)

print("\nBuilding model...")

## Designing Model
def design_model(training_data):
  
  ## Sequential()
  model = Sequential()

  ## Input Layer
  model.add(tf.keras.Input(shape=(256, 256, 1)))

  ## Hidden Convolution Layers
  model.add(layers.Conv2D(6, 6, strides=3, activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(layers.Dropout(0.2))
  model.add(layers.Conv2D(3, 3, strides=2, activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
  model.add(layers.Dropout(0.3))

  ## Additional Test Layers
  # model.add(layers.Conv2D(2, 2, strides=1, activation='relu'))
  # model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  # model.add(layers.Dropout(0.1))

  ## Flatten Layer
  model.add(layers.Flatten())

  ## Output Layer
  model.add(layers.Dense(3, activation='softmax'))

  print("\nCompiling model...")
  ## Compile model
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])
  
  ## Summmary
  model.summary()
  
  ## Return model
  return model

## Call model on training iterator
model = design_model(training_iterator)

## Early Stopping to prevent overfitting
es = EarlyStopping(monitor='val_auc', mode='min', verbose=1, patience=30)

## Using .fit() to train model
print("\nTraining model...")
history = model.fit(training_iterator, steps_per_epoch=training_iterator.samples / BATCH_SIZE, epochs=5, validation_data=validation_iterator, validation_steps=validation_iterator.samples / BATCH_SIZE, callbacks=[es])


## Figure of model for our metrics (categorical accuracy and AUC)
fig = plt.figure()

## Subplot 1 is for categorical accuracy
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

## Subplot 2 is for AUC
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.set_legend(['train', 'validation'], loc='upper left')

fig.tight_layout()

## saves figure as image
fig.savefig('static/images/my_plots.png')

plt.show()


## Classification Report
test_steps_per_epoch = numpy.math.ceil(validation_iterator.samples / validation_iterator.batch_size)

predictions = model.predict(validation_iterator, steps=test_steps_per_epoch)

test_steps_per_epoch = numpy.math.ceil(validation_iterator.samples / validation_iterator.batch_size)

predicted_classes = numpy.argmax(predictions, axis=1)
true_classes = validation_iterator.classes
class_labels = list(validation_iterator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

## categorical_accuracy: 0.4444
## auc: 0.6033


## Confusion Matrix
cm=confusion_matrix(true_classes, predicted_classes)
print(cm)

## Model + Reporting Complete //