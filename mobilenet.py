#https://hackernoon.com/efficient-implementation-of-mobilenet-and-yolo-object-detection-algorithms-for-image-annotation-717e867fa27d
from keras import applications
from keras.utils import to_categorical
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import load_model
import os
import re
import pickle
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import matplotlib.pyplot as plt 

image_width, image_height= 256, 256
rootdir = "/media/sf_Runes/test"

nb_train_samples = 5
n_classes = 5
nb_validation_sample = 5
batch_size = 32


model = applications.MobileNetV2(weights= "imagenet", include_top=False, input_shape=(image_height, image_width,3))
x = GlobalAveragePooling2D()(model.output)
x=Dense(96, activation="relu")(x)
x=Dropout(0.5)(x)
predictions = Dense(5, activation="softmax")(x)
model =Model(input=model.input, output=predictions)
for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True

model.compile(loss="categorical_crossentropy", optimizer=optimizers.nadam(lr=0.00001), metrics=["accuracy"])


train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    rootdir,
    target_size=(256,256),
    batch_size=batch_size,
    class_mode='categorical',
    save_to_dir='/home/jasmin/Projects/test_folder',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    rootdir, # same directory as training data
    target_size=(256,256),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data

for i in range(1):
    gen_data.next()
#es = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
#model.fit_generator(train_generator,validation_data = validation_generator, epochs = 1, verbose=1,callbacks=[es])
# serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")