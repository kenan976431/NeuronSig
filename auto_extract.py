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

for j in range(43):
    os.system('python feature_extract.py --batch {}'.format(j))

