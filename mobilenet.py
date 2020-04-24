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
rootdir = "/media/sf_Runes"

nb_train_samples = 5
n_classes = 5
nb_validation_sample = 5
batch_size = 8


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

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode="nearest",
                                   width_shift_range=0.3,
                                   height_shift_range=0.3,
                                   rotation_range=30)

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
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    rootdir, # same directory as training data
    target_size=(256,256),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data

model.fit_generator(train_generator,validation_data = validation_generator, epochs = 1, verbose=1)
#print(model.summary())
#uncomment the follwoing to save your weights and model.
'''model_json=model_final.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)
model_final.save_weights("weights_VGG.h5")
model_final.save("model_27.h5")
#model_final.predict(test_set, batch_size=batch_size)
'''
'''
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights_VGG.h5",by_name=True)
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

#print(loaded_model.summary())
loaded_model.fit_generator(training_set,                         steps_per_epoch = 1000,epochs = 100,                         validation_data = test_set,validation_steps=1000)
#score = loaded_model.evaluate(training_set,test_set , verbose=0)
'''