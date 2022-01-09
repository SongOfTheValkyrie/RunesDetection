#https://hackernoon.com/efficient-implementation-of-mobilenet-and-yolo-object-detection-algorithms-for-image-annotation-717e867fa27d
import tensorflow as tf
from tensorflow import keras
from keras import applications
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix,classification_report
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
rootdir = "C:/Users/Jasmin/Documents/RunesApp/Runes"
batch_size = 32

model = applications.mobilenet_v2.MobileNetV2(weights= "imagenet", include_top=False, input_shape=(image_height, image_width,3))
x = GlobalAveragePooling2D()(model.output)
x=Dense(96, activation="relu")(x)
x=Dropout(0.5)(x)
predictions = Dense(5, activation="softmax")(x)
for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True

adam=tf.keras.optimizers.Adam(
    learning_rate=0.00001, amsgrad=False,
    name='Adam'
)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    validation_split=0.2) # set validation split

eval_datagen = ImageDataGenerator(rescale=1./255) # set validation split

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


eval_generator = eval_datagen.flow_from_directory(
    rootdir,
    target_size=(256,256),
    batch_size=1,
    class_mode='categorical',
    shuffle=False) # set as evaluation data

es = EarlyStopping(monitor='val_loss',min_delta=0.0001, patience=15, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
#model.fit_generator(train_generator,validation_data = validation_generator, epochs = 1, verbose=1,callbacks=[es])
# serialize weights to HDF5
model.save_weights("modeladam.h5")
print("Saved model to disk")

print('\n# Evaluate on test data')
results = model.evaluate_generator(generator=eval_generator)
print('test loss, test acc:', results)

Y_pred = model.predict_generator(generator=eval_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(eval_generator.classes, y_pred))
y_test = eval_generator.classes
target_names = ['Algiz', 'Fehu', 'Ingwaz','Sowilo']
print(classification_report(eval_generator.classes, y_pred, target_names=target_names))

#cm = confusion_matrix(y_test, y_pred)
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#cm.diagonal()
#acc_each_class = cm.diagonal()

#print('accuracy of each class: \n')
#for i in range(len(labels)):
#  print(labels[i], ' : ', acc_each_class[i])
#print('\n')