from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense
import numpy as np
import matplotlib.pyplot as plt
import os

dataset = ImageDataGenerator()
train = dataset.flow_from_directory('DATASET/train',
                                    class_mode='categorical')
test = dataset.flow_from_directory('DATASET/test',
                                    class_mode='categorical')
print(train.class_indices)

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (256, 256, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), input_shape = (256, 256, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), input_shape = (256, 256, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#print(model.summary())
#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=(200, 1000)),
#    keras.layers.Dense(50, activation=tf.nn.relu),
#    keras.layers.Dense(4, activation=tf.nn.softmax)
#])

result = model.fit_generator(train, epochs=1, steps_per_epoch=50,
          validation_data=test, validation_steps= 12 )

print(result.history["acc"])
#print(tf.__version__)

#fashion_mnist = keras.datasets.fashion_mnist

#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
#              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

#train_images = train_images / 255.0

#test_images = test_images / 255.0

# plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
# plt.show()

#model = keras.Sequential([
#    keras.layers.Flatten(input_shape=(200, 1000)),
#    keras.layers.Dense(50, activation=tf.nn.relu),
#    keras.layers.Dense(4, activation=tf.nn.softmax)
#])

#model.compile(optimizer='adam', 
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])

#model.fit(train_images, train_labels, epochs=5)

#test_loss, test_acc = model.evaluate(test_images, test_labels)
#
#print('Test accuracy:', test_acc)
#
#def plot_image(i, predictions_array, true_label, img):
#  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
#  plt.grid(False)
#  plt.xticks([])
#  plt.yticks([])
#  
#  plt.imshow(img, cmap=plt.cm.binary)
#
#  predicted_label = np.argmax(predictions_array)
#  if predicted_label == true_label:
#    color = 'blue'
#  else:
#    color = 'red'
#  
#  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                100*np.max(predictions_array),
#                                class_names[true_label]),
#                                color=color)
#
#predictions = model.predict(test_images)
#
#def plot_value_array(i, predictions_array, true_label):
#  predictions_array, true_label = predictions_array[i], true_label[i]
#  plt.grid(False)
#  plt.xticks([])
#  plt.yticks([])
#  thisplot = plt.bar(range(10), predictions_array, color="#777777")
#  plt.ylim([0, 1]) 
#  predicted_label = np.argmax(predictions_array)
#
#  thisplot[predicted_label].set_color('red')
#  thisplot[true_label].set_color('blue')
#
#num_rows = 5
#num_cols = 3
#num_images = num_rows*num_cols
#plt.figure(figsize=(2*2*num_cols, 2*num_rows))
#for i in range(num_images):
#  plt.subplot(num_rows, 2*num_cols, 2*i+1)
#  plot_image(i, predictions, test_labels, test_images)
#  plt.subplot(num_rows, 2*num_cols, 2*i+2)
#  plot_value_array(i, predictions, test_labels)
#plt.show()