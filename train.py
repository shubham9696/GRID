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



conv_kwargs = dict(
    padding='same',
)

pool_kwargs = dict(
    pool_size=2,
)


image_input = tf.keras.Input(shape=(480, 640, 3), name='input_layer')
conv_1 = tf.keras.layers.Conv2D(filters=16,kernel_size=(8, 8),padding='same')(image_input)
l1=tf.keras.layers.LeakyReLU()(conv_1)
conv_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(l1)

conv_2 = tf.keras.layers.Conv2D(filters=32,kernel_size=(4, 4),strides=(2,2),padding='same')(conv_1)
conv_2=tf.keras.layers.LeakyReLU()(conv_2)
conv_2 = tf.keras.layers.Conv2D(filters=32,kernel_size=(4, 4),padding='same')(conv_2)
conv_2=tf.keras.layers.LeakyReLU()(conv_2)
conv_2 = tf.keras.layers.Conv2D(filters=32,kernel_size=(4, 4),padding='same')(conv_2)
conv_2=tf.keras.layers.LeakyReLU()(conv_2)

conv_3 = tf.keras.layers.Conv2D(filters=64,kernel_size=(4, 4),strides=(2,2),padding='same')(conv_2)
conv_3=tf.keras.layers.LeakyReLU()(conv_3)
conv_3 = tf.keras.layers.Conv2D(filters=64,kernel_size=(4, 4),padding='same')(conv_3)
conv_3=tf.keras.layers.LeakyReLU()(conv_3)
conv_3 = tf.keras.layers.Conv2D(filters=64,kernel_size=(4, 4),padding='same')(conv_3)
conv_3=tf.keras.layers.LeakyReLU()(conv_3)

conv_4 = tf.keras.layers.Conv2D(filters=128,kernel_size=(4, 4),strides=(2,2),padding='same')(conv_3)
conv_4=tf.keras.layers.LeakyReLU()(conv_4)
conv_4 = tf.keras.layers.Conv2D(filters=128,kernel_size=(4, 4),padding='same')(conv_4)
conv_4=tf.keras.layers.LeakyReLU()(conv_4)
conv_4 = tf.keras.layers.Conv2D(filters=128,kernel_size=(4, 4),padding='same')(conv_4)
conv_4=tf.keras.layers.LeakyReLU()(conv_4)

conv_5 = tf.keras.layers.Conv2D(filters=256,kernel_size=(4, 4),strides=(2,2),padding='same')(conv_4)
conv_5=tf.keras.layers.LeakyReLU()(conv_5)
conv_5 = tf.keras.layers.Conv2D(filters=128,kernel_size=(4, 4),padding='same')(conv_5)
conv_5=tf.keras.layers.LeakyReLU()(conv_5)
conv_5 = tf.keras.layers.Conv2D(filters=128,kernel_size=(4, 4),padding='same')(conv_5)
conv_5=tf.keras.layers.LeakyReLU()(conv_5)

avg_f=tf.keras.layers.AvgPool2D(name="gp",pool_size=(4,4))(conv_5)

conv_flat = tf.keras.layers.Flatten()(avg_f)
fc1=tf.keras.layers.Dense(64,name="fc1")(conv_flat)
drop1=tf.keras.layers.Dropout(name="Drop1",rate=0.3)(fc1)

fc2=tf.keras.layers.Dense(64,name="fc2")(drop1)
fc22=tf.keras.layers.Dropout(name="Drop2",rate=0.3)(fc2)

out=tf.keras.layers.Dense(4,name="out")(fc2)

model = tf.keras.Model(inputs=image_input, outputs=[out])
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.0001),loss="mse",metrics=[IoU])

# data=np.zeros((N,480,640,3))
# label=[]
# map={}
# cnt=0
#
# image_paths=sorted(list(train["image_name"]))
# #print(image_paths)
# random.seed(42)
# random.shuffle(image_paths)
#
# for i in range(N):
#     map[train.iloc[i,0]]=np.array(train.iloc[i,1:4])
#
# for img in image_paths:
#     image=cv2.imread("train_images/"+img)
#     data[cnt]=img_to_array(image)
#     cnt+=1
#     print(cnt)
#     label.append(map[img])
#
# data=np.array(data,dtype="float")
# label=np.array(label,dtype="float")

def load_image(image_path):
    # data augmentation logic such as random rotations can be added here
    return img_to_array(load_img(image_path))

class custom_sequence(tf.keras.utils.Sequence):
    def __init__(self,df_path,data_path,batch_size,mode="train"):
        self.df=pd.read_csv(df_path)
        self.batch_size=batch_size
        self.mode=mode

        self.x=[list(self.df.iloc[i,1:5]) for i in range(len(self.df["image_name"]))]
        self.image_list = self.df["image_name"].apply(lambda x: os.path.join(data_path,x)).tolist()

    def __len__(self):
        return int(math.ceil(len(self.df)/float(self.batch_size)))

    def on_epoch_end(self):
        self.indexes=range(len(self.image_list))
        if self.mode=='train':
            self.indexes=random.sample(self.indexes,k=len(self.indexes))

    def get_batch_labels(self,idx):
        label=[self.x[idx*self.batch_size:(1+idx)*self.batch_size]]
        return label

    def get_batch_features(self,idx):
        batch_images=self.image_list[idx*self.batch_size:(1+idx)*self.batch_size]
        return np.array([load_image(im) for im in batch_images])

    def __getitem__(self, idx):
        batch_x=self.get_batch_features(idx)
        batch_y=self.get_batch_labels(idx)
        return batch_x,batch_y



seq = custom_sequence('./training.csv',
                       './train_images/',
                       batch_size=16)


callbacks = [
    tf.keras.callbacks.ModelCheckpoint('./model.h5', verbose=1)
]

if __name__ == '__main__':
    model.fit_generator(generator=seq,verbose=1,epochs=5,use_multiprocessing=True,workers=10,callbacks=callbacks)



