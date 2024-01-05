import os
import numpy as np
import random
import os
import cv2
import math
import numpy.linalg as la
import shutil
import keras
import argparse
from keras.datasets.cifar10 import load_data
from PIL import Image
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt
from art.estimators.classification import KerasClassifier
import tensorflow as tf
from keras.utils import to_categorical

# adv = np.load('/public/lixiaohao/code/dataset/cifar10/VGG19/adv/JSMA/train/adv_test_x.npy')
# print(len(adv))
# batch_max = len(adv) / 50
# print(batch_max)
# noise = [0.01, 0.02, 0.03, 0.05, 0.1, 0.12, 0.13, 0.15, 0.2]
# for i in noise:
# os.system('source activate py36_lxh')
for j in range(43):
    os.system('python feature_extract.py --batch {}'.format(j))
