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
import argparse
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
tf.compat.v1.disable_eager_execution()

# %%
parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='normal')
parser.add_argument('--batch', type=int, default=0)
# parser.add_argument('--noise_init', type=float, default=0.01)
args = parser.parse_args()
# batch_size = []
# %%
# 模型、对抗样本载入
model = load_model('/data0/jinhaibo/DGAN/train_model/resnet20.h5')
# model.summary()
# %%GTSRB
x_train = np.load('/data0/jinhaibo/DGAN/GTSRB_img_Crop/Final_Training/GTSRB100.npy')
y_train = np.load('/data0/jinhaibo/DGAN/GTSRB_img_Crop/Final_Training/GTSRB100-labels.npy')
y_train = to_categorical(y_train, 43)
x_train = x_train /255.0
x_train_1 = []
for i in range(43):
    x = x_train[np.argmax(y_train, axis=1)==i]
    x_train_1.extend(x[:50])
advs = np.load('/data0/jinhaibo/DGAN/adv_GTSRB/resnet20/AUNA/adv.npy')
advs_label = np.load('/data0/jinhaibo/DGAN/adv_GTSRB/resnet20/AUNA/truth_label.npy')
advs_label = to_categorical(advs_label, 43)
adv_train = []
# adv_test = []
for i in range(43):
    x = advs[np.argmax(advs_label, axis=1)==i]
    adv_train.extend(x[:50])
x_train = np.array(x_train_1)
x_adv = np.array(adv_train)
#%%
# (x_train, y_train), (x_test, y_test) = load_data()
# x_train = x_train / 255.0
# x_test = x_test / 255.0
# y_train = to_categorical(y_train, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)
# batch1 = args.batch
# print(batch1)
# path = '/data0/jinhaibo/lixiaohao/Adaptive/cifar10_VGG19/adv_x' + str(batch1)+'.npy'
# print(path)
# advs = np.load(path)
# advs = []
# for i in range(2):
#     path = '/data0/jinhaibo/lixiaohao/adv_cifar10/Vgg16/Boundary/adv_train_' + str(i+1) + '.npy'
#     advs.extend(np.load(path))
# advs_train = []
# advs_test = []
# for i in range(10):
#     path = '/data0/jinhaibo/DGAN/adv_imagenet/vgg/adv_examples/JSMA/adv_test_x_' + str(i)+'.npy'
#     adv = np.load(path)
#     advs_train.extend(adv[:20])
#     advs_test.extend(adv[20:50])
# advs = np.vstack((advs_train, advs_test))
#
# x_train = np.load('/data0/jinhaibo/DGAN/animals_10_datasets/vgg/train/img_data.npy')
# x_train = x_train/255.0
# y_train = np.load('/data0/jinhaibo/DGAN/animals_10_datasets/vgg/train/img_data_label.npy')
# %%
#### scale neuron values
names_layer = 'flatten'


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


#### only choose fc2 to cal.


def get_activation_values(input_data, model):
    layer_names = [layer.name for layer in model.layers if names_layer in layer.name]
    get_value = [[] for j in range(len(layer_names))]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output)
        for num_neuron in range(scaled.shape[-1]):
            get_value[i].append(np.mean(scaled[..., num_neuron]))
    return get_value


def init_position_tables(model):
    model_layer_dict = defaultdict(bool)
    init_dict(model, model_layer_dict)
    return model_layer_dict


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if names_layer in layer.name:
            for index in range(layer.output_shape[-1]):
                model_layer_dict[(layer.name, index)] = False


def get_position(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if names_layer in layer.name]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output)
        for num_neuron in range(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True


# %%
layer_num = 64  # #fc2 total neurons
benign_features = []
# num_sample = 300  # # expend the data
# batch1 = args.batch
batch1 = args.batch
print(batch1)
batch = batch1 * 50
# for i in range(args.batch * 50, (args.batch + 1) * 50):
#     x = x_train[i:i + 1]
#     model_layer_dict = init_position_tables(model)
#     print(i)
#     neuron_values = get_activation_values(x, model)
#     get_position(x, model, model_layer_dict, threshold=0.5)
#     # ## total neuron = 4096
#     not_activated = [(layer_name, index) for (layer_name, index), v in list(model_layer_dict.items())[:layer_num] if
#                      not v]
#     neuron_positions = [0 if not v else 1 for (layer_name, index), v in list(model_layer_dict.items())[:layer_num]]
#     features = np.hstack((np.array(neuron_values).reshape(-1), np.array(neuron_positions).reshape(-1))).reshape(1, -1)
#     benign_features.append(features)
# batch1 = args.batch
adv_features = []
for i in range(50):
    x = x_adv[i + batch: i + batch + 1]
    model_layer_dict = init_position_tables(model)
    print(i)
    neuron_values = get_activation_values(x, model)
    # print(neuron_values)
    get_position(x, model, model_layer_dict, threshold=0.5)
    # ## total neuron = 4096
    not_activated = [(layer_name, index) for (layer_name, index), v in list(model_layer_dict.items())[:layer_num] if
                     not v]
    neuron_positions = [0 if not v else 1 for (layer_name, index), v in list(model_layer_dict.items())[:layer_num]]
    features = np.hstack((np.array(neuron_values).reshape(-1), np.array(neuron_positions).reshape(-1))).reshape(1, -1)
    adv_features.append(features)
path1 = '/data0/jinhaibo/lixiaohao/adv_GTSRB/ResNet20/noise'
if not os.path.exists(path1):
    os.makedirs(path1)
print(path1)
np.save(path1+'/feature_'+str(batch1) + '.npy', adv_features)
