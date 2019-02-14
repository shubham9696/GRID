import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import cv2
import os
import math
import argparse
import random
from imutils import paths

test = pd.read_csv("test.csv")
test.head()
test_set=list(test["image_name"])

train = pd.read_csv("training.csv")
train.head()
N=len(list(train["image_name"]))


def calculate_iou(y_true, y_pred):
    """
    Input:
    Keras provides the input as numpy arrays with shape (batch_size, num_columns).

    Arguments:
    y_true -- first box, numpy array with format [x, y, width, height, conf_score]
    y_pred -- second box, numpy array with format [x, y, width, height, conf_score]
    x any y are the coordinates of the top left corner of each box.

    Output: IoU of type float32. (This is a ratio. Max is 1. Min is 0.)

    """

    results = []
    for i in range(0, y_true.shape[0]):
        # set the types so we are sure what type we are using
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)

        # boxTrue
        x_boxTrue_tleft = y_true[i, 0]  # numpy index selection
        y_boxTrue_tleft = y_true[i, 2]
        x_boxTrue_br = y_true[i, 1]
        y_boxTrue_br = y_true[i, 3] # Version 2 revision

        # boxPred
        x_boxPred_tleft = min(y_pred[i, 0],y_pred[i,1])
        y_boxPred_tleft = min(y_pred[i, 2],y_pred[i,3])
        x_boxPred_br = max(y_pred[i, 0],y_pred[i,1])
        y_boxPred_br = max(y_pred[i,2],y_pred[i,3])


        boxTrue_width = x_boxTrue_br-x_boxTrue_tleft
        boxTrue_height = y_boxTrue_br-y_boxTrue_tleft
        area_boxTrue = (boxTrue_width * boxTrue_height)

        boxPred_width = x_boxPred_br-x_boxPred_tleft
        boxPred_height = y_boxPred_br-y_boxPred_tleft
        area_boxPred = (boxPred_width * boxPred_height)

        # calculate the bottom right coordinates for boxTrue and boxPred

        # boxTrue
         # Version 2 revision

        # calculate the top left and bottom right coordinates for the intersection box, boxInt

        # boxInt - top left coords
        x_boxInt_tleft = np.max([x_boxTrue_tleft, x_boxPred_tleft])
        y_boxInt_tleft = np.max([y_boxTrue_tleft, y_boxPred_tleft])  # Version 2 revision

        # boxInt - bottom right coords
        x_boxInt_br = np.min([x_boxTrue_br, x_boxPred_br])
        y_boxInt_br = np.min([y_boxTrue_br, y_boxPred_br])

        # Calculate the area of boxInt, i.e. the area of the intersection
        # between boxTrue and boxPred.
        # The np.max() function forces the intersection area to 0 if the boxes don't overlap.


        # Version 2 revision
        area_of_intersection = \
            np.max([0, (x_boxInt_br - x_boxInt_tleft)]) * np.max([0, (y_boxInt_br - y_boxInt_tleft)])

        iou = area_of_intersection / ((area_boxTrue + area_boxPred) - area_of_intersection)

        # This must match the type used in py_func
        iou = iou.astype(np.float32)

        # append the result to a list at the end of each loop
        print(iou)
        results.append(iou)

    # return the mean IoU score for the batch
    return np.mean(results)


def IoU(y_true, y_pred):
    # Note: the type float32 is very important. It must be the same type as the output from
    # the python function above or you too may spend many late night hours
    # trying to debug and almost give up.

    iou = tf.py_func(calculate_iou, [y_true, y_pred], tf.float32)

    return iou


test_gen=ImageDataGenerator()

test_seq=test_gen.flow_from_directory("./test",target_size=(480,640),batch_size=1,shuffle=False)
im_name=list(test_seq.filenames)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('./model.h5', verbose=1)
]

if __name__ == '__main__':
    model=tf.keras.models.load_model("./model.h5")
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.0001), loss="mse", metrics=[IoU])
    prediction=model.predict_generator(test_seq,verbose=1,workers=0,use_multiprocessing=False)
    for i in range(len(prediction)):
        print(im_name[i])
        print("::::")
        print(min(prediction[i,0],prediction[i,2]))
        print(min(prediction[i, 1], prediction[i, 3]))
        print(max(prediction[i, 0], prediction[i, 2]))
        print(max(prediction[i, 1], prediction[i, 3]))


