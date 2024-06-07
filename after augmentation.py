import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
dataset_link='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz' #URL of the dataset source
string_data_directory = tf.keras.utils.get_file('flower_photos', origin=dataset_link, untar=True)      #downloads the file from given URL
data_directory = pathlib.Path(string_data_directory)                                                   #path of the dataset

img_height,img_width=180,180 
batch_size=32
training_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_directory,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)    #creates a tensorflow dataset from the given directory #creates training_dataset


validation_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_directory,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)  #creates validation dataset


class_names = training_data.class_names #creates variables for the classes of the dataset
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))    #plots first 9 images of the training_dataset
for images, labels in training_data.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

num_classes = 5

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

model1 = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),  #rescales input values between [0,1] without changing the shape
  layers.Conv2D(512, 3, padding='same', activation='relu'), #convolution layer with 512 filters
  layers.MaxPooling2D(),                                    #takes max value from the feature map
  layers.Conv2D(256, 3, padding='same', activation='relu'), #convolution layer with 256 filters
  layers.MaxPooling2D(),                                    #takes max value from the above feature map
  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),                             #drops 20% of neurons
  layers.Flatten(),                                #flattens the feature map into a 1D vector
  layers.Dense(128, activation='relu'),
  layers.Dense(64,activation='relu'),
  layers.Dense(32,activation='relu'),
  layers.Dense(num_classes, activation='softmax')
])
model1.compile(optimizer='adam',               #adam optimizer adjusts the laearning rate
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),#gives the loss of training_dataset and validation_dataset
              metrics=['accuracy'])            #applies softmax activation for accuray

epochs = 40
history = model1.fit(
  training_data,
  validation_data=validation_data,
  epochs=epochs
)

training_acc = history.history['accuracy']
validation_acc = history.history['val_accuracy']

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, training_acc, color='red',label='Training Accuracy')
plt.plot(epochs_range, validation_acc,color='green', label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, training_loss, color='red', label='Training Loss')
plt.plot(epochs_range, validation_loss,color='green', label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model1.save('with_augmentation.h5')