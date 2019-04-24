import tensorflow as tf
import keras
from keras import models
from keras.utils import multi_gpu_model
import numpy as np
from keras.backend.tensorflow_backend import set_session

keras.backend.set_image_data_format('channels_first')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

NUM_EPOCHS = 3
CHANNEL = 3
NUM_CLASSES = 1000
NUM_SAMPLES = 2048

HEIGHT = 900
WIDTH = 660
BATCH_SIZE = 1
CHANNEL_BASE = 512

x = np.random.random((NUM_SAMPLES, CHANNEL, HEIGHT, WIDTH))
y = np.random.random((NUM_SAMPLES, NUM_CLASSES))


def my_cnn():
  model = models.Sequential()
  model.add(keras.layers.Conv2D(CHANNEL_BASE, 3, activation='relu',
                                input_shape=(CHANNEL, HEIGHT, WIDTH)))
  model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))
  model.add(keras.layers.Conv2D(CHANNEL_BASE * 2, 3, activation='relu'))
  model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))
  model.add(keras.layers.Conv2D(CHANNEL_BASE * 4, 3, activation='relu'))
  model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))
  model.add(keras.layers.Conv2D(CHANNEL_BASE, 3, activation='relu'))
  model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))
  model.add(keras.layers.Reshape((-1,)))
  model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
  return model


# Single GPU
model = my_cnn()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('')
print('Single-GPU ..........................................')
model.fit(x, y,
          epochs=NUM_EPOCHS,
          batch_size=BATCH_SIZE)