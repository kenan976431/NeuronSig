from art.defences.preprocessor import FeatureSqueezing
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from art.attacks.evasion import FastGradientMethod
import numpy as np
from sklearn.metrics import roc_auc_score
import math
import time
from keras.models import load_model
import os
import tensorflow as tf
from keras.datasets import cifar10
from art.estimators.classification import KerasClassifier
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.compat.v1.disable_eager_execution()
# model = load_model('/home/lxh/lixiaohao/code/dataset/cifar10/alexnet/cifar10_alexnet.h5')
model = load_model('/data0/jinhaibo/DGAN/Inverse_Peturbation/Voiceprint_nip/models/DeepSpeaker_all.h5')
# %%VCTK
def get_data_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        strings = f.readlines()
        audiolist = np.array([string.split('\t')[0] for string in strings])
        labellist = np.array([int(string.split('\t')[1]) for string in strings])
        namelist = np.array([string.split("\t")[0].split("/")[-1].split(".npz")[0] for string in strings])
        f.close()
    return audiolist, labellist, namelist


wav_root = "/data0/BigPlatform/ZJPlatform/001_Audio/000-Dataset/000-Voice/001-Vocal_set/000-VCTK/VCTK_old/Same_length/WAV"
train_list = "/data0/BigPlatform/ZJPlatform/001_Audio/001-Demo/Voiceprint_GJ/VCTK_data_lists/train_list.txt"
npz_paths, label_lists, name_lists = get_data_list(train_list)
npz_path = npz_paths  # 取1000个数据
label_list = label_lists
name_list = name_lists
Y = label_list
X = [np.load(npz)["coded_sp"] for npz in npz_path]
x_train = np.array(X)
y_train = to_categorical(Y, 30)
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = x_train/255.0
# x_test = x_test/255.0
# y_train = to_categorical(y_train, 10)
# x_train = np.load('/data0/jinhaibo/DGAN/animals_10_datasets/vgg/train/img_data.npy')
#
# y_train = np.load('/data0/jinhaibo/DGAN/animals_10_datasets/vgg/train/img_data_label.npy')
# advs_train = []
# advs = []
# for i in range(5):
#     path = '/data0/jinhaibo/lixiaohao/adv_cifar10/Vgg16/PGD/adv_train_' + str(i+1)+'.npy'
#     adv = np.load(path)
#     if i== 0:
#         advs = adv
#     else:
#         advs = np.vstack((advs, adv))
#
#%%
advs = np.load('/data0/jinhaibo/DGAN/Inverse_Peturbation/Voiceprint_nip/DeepSpeaker/Boundary/advs.npy')
# advs = advs[:, 0, :, :, :]
# for i in range(500):
    # y_predict = model.predict(x_train[i: i+1])
    # print(np.argmax(y_predict), np.argmax(y_train[i]))



#%%
# start = time.time()
classifier = KerasClassifier(model=model, clip_values=(0, 1))
x_train_adv = advs[:400]
x_train_clean = x_train[:400]
benign_labels = np.array([0 for i in range(400)]).reshape(400, -1)
adv_labels = np.array([1 for i in range(400)]).reshape(400, -1)
x_test_hun = np.vstack((x_train_adv[:400], x_train[:400]))
y_test_hun = np.vstack((adv_labels, benign_labels))
y_test_hun = to_categorical(y_test_hun, 2)
fs = FeatureSqueezing(clip_values=(0, 1))
x_defense = fs(x_test_hun)

#%%
y_predict = []
y_defense = model.predict(x_defense[0])
y_ori = model.predict(x_test_hun)
for i in range(800):
    if np.argmax(y_defense[i]) == np.argmax(y_ori[i]):
        y_predict.append(0)
    else:
        y_predict.append(1)
TPR = 0
FPR = 0
for i in range(400):
    if y_predict[i] == 1:
        TPR += 1
    if y_predict[i+400] == 0:
        FPR += 1
print(TPR/400)
print(FPR/400)
# end = time.time()
# print(end-start)
# for i in range(len(x_defense)):
#     print(i)
#     if np.argmax(model.predict(x_defense[i:i+1])) == np.argmax(model.predict(x_test_hun[i:i+1])):
#         y_predict.append(0)
#     else:
#         y_predict.append(1)
#%%
detect_pred_test = to_categorical(y_predict, 2)
roc_value = roc_auc_score(y_test_hun, detect_pred_test)
print('roc_value', roc_value)

