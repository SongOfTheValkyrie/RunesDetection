from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import cv2
import tensorflow.compat.v1 as tf
from datasets import build_dataset
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from mobilenetv3 import MobileNetV3

dataset_path="C:/Users/Jasmin/Documents/RunesApp/Runes"
tf.disable_v2_behavior() 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

train_images_x = []
y=[]
x=[]
folder_list=[]

def load_dataset():
    count=0    
    #fill rune classes
    for image in os.walk(dataset_path):
        if count==0:
            folder_list=image[1]
        train_images_x.append(image[2]) 
        for i in train_images_x[count]:
            path=(str(dataset_path)+"/"+str(folder_list[count-1])+"/"+str(i))
            img = cv2.imread(path ,cv2.IMREAD_UNCHANGED)
            if img is None:
                print("there is broken data:" ,path)
            else:
                dim = (224,224)
                # resize image
                img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                img = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
                img = np.expand_dims(img, axis=0)
                img = img/255
                x.append(img)
                y.append(count-1)
            
        count+=1

    print("Succesfully load and prepared Dataset!")
    return x,y
    
    

def build_dataset():
    train_X, test_X, train_y, test_y = train_test_split(x, y, 
                                                    train_size=0.6,
                                                    test_size=0.4)
   

    return train_X,test_X,train_y,test_y


_available_optimizers = {
    "rmsprop": tf.keras.optimizers.RMSprop(),
    "adam": tf.keras.optimizers.Adam(),
    "sgd": tf.keras.optimizers.SGD(),
    }

model = MobileNetV3((224,224,3),4).build()


model.compile(
        optimizer=_available_optimizers.get("sgd"),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

x,y=load_dataset()

x_train,x_test, x_eval,y_eval=build_dataset()


model.fit(
        x_train,
        x_test,
        batch_size=10,
        epochs=2,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_eval, y_eval)
)

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')