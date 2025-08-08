# %% use tf2.x
from keras.layers import Input
from keras.datasets.cifar10 import load_data
from keras.models import load_model
import numpy as np
from keras.utils import to_categorical
from keras import backend as K
import os
from keras.models import Model
import tensorflow as tf
from keras.layers import Input, Dense
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.compat.v1.disable_eager_execution()

# %%
model = load_model('adv_cifar10/Vgg16/VGG16.h5')
model.summary()

def init_coverage_tables(model1):
    model_layer_dict1 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    return model_layer_dict1

def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


model_layer_dict1 = init_coverage_tables(model)


aaa = len(model_layer_dict1)


#%%
(x_train, y_train), (x_test, y_test) = load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# %%
benign_features = []
adv_features = []
# path = 'features4demo/CIFAR10/VGG19/FGSM/Adv'
for i in range(10):
    path_Benign = 'dv_cifar10/Vgg16/clean/' + str(i) + '.npy'
    benign_features.extend(np.load(path_Benign))
    path_adv = 'adv_cifar10/Vgg16/FGSM/' + str(i) + '.npy'
    adv_features.extend(np.load(path_adv))
# data = np.array(benign_features)
# print(data.shape)
# %%
test_features = []
path = 'adv_cifar10/Vgg16/JSMA/'
for i in range(10):
    path_adv = path + '/' + str(i) + '.npy'
    test_features.extend(np.load(path_adv))
    # print(len(np.load(path_adv)))
# print(len(test_features))

# %%
num_sample = 400
benign_features = np.array(benign_features[:num_sample]).reshape(num_sample, -1)
adv_features = np.array(adv_features[:num_sample]).reshape(num_sample, -1)
benign_labels = np.array([0 for i in range(num_sample)]).reshape(num_sample, -1)
adv_labels = np.array([1 for i in range(num_sample)]).reshape(num_sample, -1)

test_num = 500
test_features = np.array(test_features[:]).reshape(test_num, -1)
test_labels = np.array([1 for i in range(test_num)]).reshape(test_num, -1)
# print(np.isnan(adv_labels))
# %%
sec_num = 200
x_to_train = np.vstack((benign_features[:sec_num], adv_features[:sec_num]))
y_to_train = np.vstack((benign_labels[:sec_num], adv_labels[:sec_num]))
# x_to_test = np.vstack((adv_features[sec_num:]))
# y_to_test = np.vstack((adv_labels[sec_num:]))
y_to_train_one = to_categorical(y_to_train, 2)
# y_to_test_one = to_categorical(y_to_test, 2)
x_to_test = np.vstack((test_features[:]))
y_to_test = np.vstack((test_labels[:]))
y_to_test_one = to_categorical(y_to_test, 2)
x_to_test_auc = np.vstack((test_features[sec_num:], benign_features[sec_num:]))
y_to_test_auc = np.vstack((test_labels[sec_num:], benign_labels[sec_num:]))
y_to_test_auc_one = to_categorical(y_to_test_auc, 2)

#%%
def meta_classify(input_shape):
    # x = Dense(1024, name='predictions', activation='relu')(input_shape)
    x = Dense(2, activation='softmax')(input_shape)
    model = Model(input_shape, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model_detect = meta_classify(input_shape=Input(shape=x_to_train.shape[1:]))
model_detect.fit(x_to_train_hhh, y_to_train_hhh, batch_size=5, epochs=30)
# model_detect.save('features4demo/CIFAR10/VGG19/cifar_fgsm_vgg19.h5')
# %%ACC score
loss, acc = model_detect.evaluate(x_to_test, y_to_test_one)
loss_benign, acc_benign = model_detect.evaluate(benign_features[sec_num:], to_categorical(benign_labels[sec_num:], 2))
print("Loss: %f; acc: %f; loss_beign: %f; acc_benign: %f" % (loss, acc, loss_benign, acc_benign))
# %% AUC score
detect_pred_test = model_detect.predict(x_to_test_auc)
roc_value = roc_auc_score(y_to_test_auc_one, detect_pred_test)
print('roc_value', roc_value)

