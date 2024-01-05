#%% use tf1.x
import sys
sys.path.append('/data0/jinhaibo/DGAN/AdvChecker/Baselines')
sys.path.append('/data0/jinhaibo/DGAN/AdvChecker')
from keras.layers import Input
from keras.datasets.cifar10 import load_data
from keras.models import load_model
import numpy as np
from keras.utils import to_categorical
from keras import backend as K
import os
from keras.models import Model
import tensorflow as tf
from collections import defaultdict
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.compat.v1.disable_eager_execution()

#%%
nb_classes = 43
x_train = np.load('/data0/jinhaibo/DGAN/GTSRB_img_Crop/Final_Training/GTSRB100.npy')
x_train = x_train/255
y_train = np.load('/data0/jinhaibo/DGAN/GTSRB_img_Crop/Final_Training/GTSRB100-labels.npy')
y_train = to_categorical(y_train, nb_classes)
model = load_model('/data0/jinhaibo/DGAN/train_model/lenet5.h5')
#%%
advs = np.load('/data0/jinhaibo/DGAN/adv_GTSRB/lenet/BIM/adv.npy')


#%%
#### scale neuron values
names_layer = 'dense_15'

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


# def get_activation_values(input_data, model):
#     layer_names = [layer.name for layer in model.layers if 'dense_15' in layer.name]
#     get_value = [[] for j in range(len(layer_names))]
#     intermediate_layer_model = Model(inputs=model.input,
#                                      outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
#     intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
#
#     for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
#         # scaled = scale(intermediate_layer_output)
#         for num_neuron in range(intermediate_layer_output.shape[-1]):
#             get_value[i].append(np.mean(intermediate_layer_output[..., num_neuron]))
#     return get_value

####

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
#%%
layer_num = 84
benign_features = []
th = 0.65
for i in tqdm(range(4000)):
    x = x_train[i:i+1]
    model_layer_dict = init_position_tables(model)
    # print(i)
    neuron_values = get_activation_values(x, model)
    get_position(x, model, model_layer_dict, threshold=th)
    ### total neuron = layer_num
    not_activated = [(layer_name, index) for (layer_name, index), v in list(model_layer_dict.items())[:layer_num] if not v]
    neuron_positions = [0 if not v else 1 for (layer_name, index), v in list(model_layer_dict.items())[:layer_num]]
    features = np.hstack((np.array(neuron_values).reshape(-1), np.array(neuron_positions).reshape(-1))).reshape(1, -1)
    benign_features.append(features)

adv_features = []
for i in tqdm(range(4000)):
    x = advs[i:i+1]
    model_layer_dict = init_position_tables(model)
    # print(i)
    neuron_values = get_activation_values(x, model)
    get_position(x, model, model_layer_dict, threshold=th)
    ### total neuron = layer_num
    not_activated = [(layer_name, index) for (layer_name, index), v in list(model_layer_dict.items())[:layer_num] if not v]
    neuron_positions = [0 if not v else 1 for (layer_name, index), v in list(model_layer_dict.items())[:layer_num]]
    features = np.hstack((np.array(neuron_values).reshape(-1), np.array(neuron_positions).reshape(-1))).reshape(1, -1)
    adv_features.append(features)

#%%
benign_features = np.array(benign_features).reshape(4000, -1)
adv_features = np.array(adv_features).reshape(4000, -1)
benign_labels = np.array([0 for i in range(4000)]).reshape(4000, -1)
adv_labels = np.array([1 for i in range(4000)]).reshape(4000, -1)
#%%
sec_num = 1000
x_to_train = np.vstack((benign_features[:sec_num], adv_features[:sec_num]))
y_to_train = np.vstack((benign_labels[:sec_num], adv_labels[:sec_num]))
x_to_test = np.vstack((benign_features[sec_num:], adv_features[sec_num:]))
y_to_test = np.vstack((benign_labels[sec_num:], adv_labels[sec_num:]))


#%%
from keras.layers import Input, Dense
def meta_classify(input_shape):
    x = Dense(1028, name='train')(input_shape)
    x = Dense(2, name='predictions', activation='softmax')(x)
    model = Model(input_shape, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
y_to_train_one = to_categorical(y_to_train, 2)
y_to_test_one = to_categorical(y_to_test, 2)

model_nic = meta_classify(input_shape=Input(shape=x_to_train.shape[1:]))
model_nic.fit(x_to_train, y_to_train_one, batch_size=5, epochs=66)
loss, acc = model_nic.evaluate(x_to_test, y_to_test_one)
#%%
model_nic = load_model('/data0/jinhaibo/lixiaohao/detect/lenet-rewrite.h5')
#%%
loss_benign, acc_benign = model_nic.evaluate(benign_features[sec_num:], to_categorical(benign_labels[sec_num:], 2))

#%%
detect_pred_test = model_nic.predict(x_to_test)
from sklearn.metrics import roc_auc_score

roc_value = roc_auc_score(y_to_test_one, detect_pred_test)
print("AUC:", roc_value)



