# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
import os
import shutil

# read the csv file using read_csv function of pandas
test = pd.read_csv("test.csv")
test.head()
test_set=list(test["image_name"])

train = pd.read_csv("training.csv")
train.head()
print(train.iloc[0])


def load_images_from_folder(folder):
    cnt = 0
    for filename in os.listdir(folder):
        if filename in test_set:
            print(filename)
            cnt+=1
            shutil.move(folder+"/"+filename,"test/"+(str(cnt)+filename))

# load_images_from_folder("images")


