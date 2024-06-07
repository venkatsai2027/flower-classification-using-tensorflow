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
string_data_directory = tf.keras.utils.get_file('flower_photos', origin=dataset_link, untar=True)   #downloads the file from given URL
data_directory = pathlib.Path(string_data_directory)  #path of the dataset

roses = list(data_directory.glob('roses/*')) #gives all the directories that contains roses/ in a given directory
print(roses[0]) #prints the path of first eklement in the list
PIL.Image.open(str(roses[0])) #prints the image with a given path

img_height,img_width=180,180
batch_size=32
training_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_directory,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size) #creates a tensorflow dataset from the given directory #creates training dataset

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_directory,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size) #creates validation dataset

class_names = training_dataset.class_names #creates variables for the classes of the dataset
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in training_dataset.take(1): #plots first 9 images of the training_dataset
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

num_classes = 5

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)), #rescales input values between [0,1] without changing the shape
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
  layers.Flatten(),                                         #flattens the feature map into a 1D vector
  layers.Dense(128, activation='relu'),                     # Create a Dense layer with 128 output units and ReLU activation
  layers.Dense(64,activation='relu'),
  layers.Dense(32,activation='relu'),
  layers.Dense(num_classes,activation='softmax')
])

model.compile(optimizer='adam',                                                     #adam optimizer adjusts the laearning rate
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #gives the loss of training_dataset and validation_dataset
              metrics=['accuracy'])                                                 #applies softmax activation for accuray
epochs=10
history = model.fit(
  training_dataset,
  validation_data=validation_dataset,
  epochs=epochs)

model.save('without_augmentation.h5')