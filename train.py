from argparse import ArgumentParser
import os
import numpy as np
import cv2
from numpy import empty
from sklearn.datasets import load_iris
import tensorflow.compat.v1 as tf
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
from mobilenetv3_factory import build_mobilenetv3
from datasets import build_dataset
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

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
            z=(str(dataset_path)+"/"+str(folder_list[count-1])+"/"+str(i))
            x.append(cv2.imread(z,cv2.IMREAD_UNCHANGED))
            y.append(count-1)
        count+=1

    print("Succesfully load Dataset!")
    
    
def data_preprocessing():
   
    count=0
    y_res=[]
    x_res=[]
    height = 224
    width = 224
    dim = (width, height)
    for img in x:
        res_img = []
        if img is None:
            print("there is broken data at index ",count)
        else:
            for i in range(len(img)-1):
                res = cv2.resize(img[i], dim, interpolation=cv2.INTER_LINEAR)
                res_img.append(res)
            x_res.append(np.array(res_img))
            y_res.append(y[count])
        count+=1
    print("Preprocessing terminated succesfully!")
    return x_res, y_res
    
    

def build_dataset():
    train_X, test_X, train_y, test_y = train_test_split(x, y, 
                                                    train_size=0.6,
                                                    test_size=0.4)
    test_X,test_y, eval_x, eval_y = train_test_split(x, y, 
                                                    train_size=0.5,
                                                    test_size=0.5)

    return train_X,test_X,train_y,test_y, eval_x,eval_y

_available_optimizers = {
    "rmsprop": tf.keras.optimizers.RMSprop(),
    "adam": tf.keras.optimizers.Adam(),
    "sgd": tf.keras.optimizers.SGD(),
    }

model = build_mobilenetv3(
    "large",
    input_shape=(224, 224, 3),
    num_classes=1001,
    width_multiplier=1.0,
)


model.compile(
        optimizer=_available_optimizers.get("sgd"),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


#model.save_weights(f"mobilenetv3_{args.model_type}_{args.dataset}_{epochs}.h5")


load_dataset()
x,y = data_preprocessing()

x_train,x_test,y_train,y_test, x_eval,y_eval=build_dataset()

img_rows=np.array(x_train[0]).shape[0]
img_cols=np.array(x_test[0]).shape[1]

X_train=np.array(x_train).reshape(np.array(x_train).shape[0],img_rows,img_cols,1)

X_test=np.array(x_test).reshape(np.array(x_test).shape[0],img_rows,img_cols,1)


Input_shape=(img_rows,img_cols,1)

model.fit(
        x_train,
        y_train,
        batch_size=10,
        epochs=2,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_eval, y_eval)
)