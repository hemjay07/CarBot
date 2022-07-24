from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow
keras = tensorflow.keras
from tensorflow.keras import layers, Sequential

from keras.preprocessing import image
#import tensorflow.compat.v1 as tfv
import tensorflow_hub as hub
import pandas as pd
import pathlib

pd.set_option("display.precision", 8)
data_root = 'Images/' 
data = pathlib.Path(data_root) 

#global graph
#graph = tfv.get_default_graph()

#with graph.as_default():
# create a data generator for traning and val
IMAGE_SHAPE = (224, 224)
#Create Augmentation
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(224, 224,3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)
  
TRAINING_DATA_DIR = str(data_root)
print(TRAINING_DATA_DIR);
datagen_kwargs = dict(rescale=1./255, validation_split=.20)
valid_datagen = image.ImageDataGenerator(**datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
TRAINING_DATA_DIR,
subset="validation",
shuffle=True,
target_size=IMAGE_SHAPE)

train_datagen = image.ImageDataGenerator(**datagen_kwargs)
train_generator = train_datagen.flow_from_directory(
TRAINING_DATA_DIR, 
subset="training",
shuffle=True,
target_size=IMAGE_SHAPE)

image_batch_train, label_batch_train = next(iter(train_generator))
print("Image batch shape: ", image_batch_train.shape)
print("Label batch shape: ", label_batch_train.shape)
dataset_labels = sorted(train_generator.class_indices.items(), key=lambda pair:pair[1])
dataset_labels = np.array([key.title() for key, value in dataset_labels])
print(dataset_labels)

model = keras.Sequential([
data_augmentation,
hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
output_shape=[1280],
trainable=False),
layers.Dropout(0.4),
layers.Dense(train_generator.num_classes, activation='softmax')
])
model.build([None, 224, 224, 3])
model.summary()
model.compile(
optimizer=keras.optimizers.Adam(),
loss='categorical_crossentropy',
metrics=['acc'])

steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)
val_steps_per_epoch = np.ceil(valid_generator.samples/valid_generator.batch_size)
hist = model.fit(
train_generator,
epochs=10,
verbose=1,
steps_per_epoch=steps_per_epoch,
validation_data=valid_generator,
validation_steps=val_steps_per_epoch).history

model.save('image_model.h5', hist)

print('Image Model saved')
