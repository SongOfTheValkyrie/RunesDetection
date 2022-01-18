from argparse import ArgumentParser
import os
import numpy as np
from numpy import empty
from sklearn.datasets import load_iris
import tensorflow.compat.v1 as tf
from skimage import io 
from mobilenetv3_factory import build_mobilenetv3
from datasets import build_dataset
from sklearn.model_selection import train_test_split

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
            x.append(io.imread(str(dataset_path)+"/"+str(folder_list[count-1])+"/"+str(i)))
            y.append(count-1)
        count+=1

    print("Succesfully load Dataset!")
    
    
def data_preprocessing():
    
    for img in x:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    print("Preprocessing terminated succesfully!")

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
data_preprocessing()
train_X,test_X,train_y,test_y, eval_x,eval_y=build_dataset()



model.fit(
        train_X,
        train_y,
        batch_size=10,
        epochs=2,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(eval_x, eval_y)
)