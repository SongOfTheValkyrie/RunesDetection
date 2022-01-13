# Copyright 2019 Bisonai Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
from argparse import ArgumentParser
import os
from numpy import empty
from sklearn.datasets import load_iris
import tensorflow.compat.v1 as tf
from mobilenetv3_factory import build_mobilenetv3
from datasets import build_dataset
from sklearn.model_selection import train_test_split

dataset_path="C:/Users/Jasmin/Documents/RunesApp/Runes"
epochs=2
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
    for image in os.walk(dataset_path):
        train_images_x.append(image[2]) 
    
    for i in range(len(train_images_x)):
        if train_images_x[i]!=empty:
            for z in train_images_x[i]:
                x.append(z)
                y.append(i-1)
                
    

def build_dataset():
    train_X, test_X, train_y, test_y = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    test_size=0.3)

_available_datasets = [
    "mnist",
    "cifar10",
    ]

_available_optimizers = {
    "rmsprop": tf.train.RMSPropOptimizer,
    "adam": tf.train.AdamOptimizer,
    "sgd": tf.train.GradientDescentOptimizer,
    }

def main(args):
    if args.dataset not in _available_datasets:
        raise NotImplementedError


""" model = build_mobilenetv3(
        args.model_type,
        input_shape=(args.height, args.width, dataset["channels"]),
        num_classes=dataset["num_classes"],
        width_multiplier=args.width_multiplier,
        l2_reg=args.l2_reg,
    )

    if args.optimizer not in _available_optimizers:
        raise NotImplementedError

    model.compile(
        optimizer=_available_optimizers.get(args.optimizer)(args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=args.logdir),
    ]

    #todo
    model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=epochs,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_val, y_val)
    )

    model.save_weights(f"mobilenetv3_{args.model_type}_{args.dataset}_{epochs}.h5")"""


if __name__ == "__main__":
    load_dataset()
    build_dataset()
    """parser = ArgumentParser()

    # Model
    parser.add_argument("--model_type", type=str, default="large", choices=["small", "large"])
    parser.add_argument("--width_multiplier", type=float, default=1.0)

    # Input
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dataset", type=str, default="mnist", choices=_available_datasets)

    # Optimizer
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="rmsprop", choices=_available_optimizers.keys())
    parser.add_argument("--l2_reg", type=float, default=1e-5)

    # Training & validation
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--valid_batch_size", type=int, default=256)
    parser.add_argument("--num_epoch", type=int, default=10)

    # Others
    parser.add_argument("--logdir", type=str, default="logdir")

    args = parser.parse_args()
    main(args)"""
