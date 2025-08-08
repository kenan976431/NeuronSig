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
model = load_model('train_model/tiny_imagenet_vgg19_10_new.h5')
model.summary()

#%%
aaa = np.load('vgg/train/img_data.npy')

#%%
# imagenet
x_train1 = np.load('vgg/train/img_data.npy')
x_train1 = x_train1/255.0
y_train1 = np.load('vgg/train/img_data_label.npy')
x_train = []
y_train = []
for i in range(10):
    if i == 0:
        x_train = x_train1[y_train1 == i][:40]
        y_train = y_train1[y_train1 == i][:40]
    else:
        x_train = np.vstack((x_train, x_train1[y_train1 == i][:40]))
        y_train = np.hstack((y_train, y_train1[y_train1 == i][:40]))
# plt.imshow(x_train[100])
# plt.show()
# %%
#### scale neuron values
names_layer = 'fc2'

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


# %%
for i in range(num_benign):
    x = x_train[i]
    x = np.expand_dims(x, axis=0)
    model_layer_dict = init_position_tables(model)
    print(i)
    neuron_values = get_activation_values(x, model)
    get_position(x, model, model_layer_dict, threshold=0.5)
    ### total neuron = 4096
    not_activated = [(layer_name, index) for (layer_name, index), v in list(model_layer_dict.items())[:layer_num] if
                     not v]
    neuron_positions = [0 if not v else 1 for (layer_name, index), v in list(model_layer_dict.items())[:layer_num]]
    if i == 0:
        benign_features = neuron_positions
    else:
        benign_features = np.add(benign_features, neuron_positions)

#%% 
x_adv = np.load('adv_imagenet/mobile/PGD/adv_x0.npy')
layer_num = 4096
neuron_positions_100 = []

for i in range(100):
    x = x_train[i]
    x = np.expand_dims(x, axis=0)
    model_layer_dict = init_position_tables(model)
    print(i)
    neuron_values = get_activation_values(x, model)
    get_position(x, model, model_layer_dict, threshold=0.5)
    ### total neuron = 4096
    not_activated = [(layer_name, index) for (layer_name, index), v in list(model_layer_dict.items())[:layer_num] if
                     not v]
    neuron_positions = [0 if not v else 1 for (layer_name, index), v in list(model_layer_dict.items())[:layer_num]]
    neuron_positions_100.append(neuron_values)
np.save('plot/Benign_activation_value_100.npy', neuron_positions_100)
#%%
# aa = np.load('plot/Benign_activation_100.npy')
#%%
# layer_num = 64  ##fc2 total neurons
benign_features = []
layer_num = 4096
# num_sample = 1000  ## expend the data size
num_benign = 400
for i in range(num_benign):
    x = x_train[i]
    x = np.expand_dims(x, axis=0)
    model_layer_dict = init_position_tables(model)
    print(i)
    neuron_values = get_activation_values(x, model)
    get_position(x, model, model_layer_dict, threshold=0.4)
    ### total neuron = 4096
    not_activated = [(layer_name, index) for (layer_name, index), v in list(model_layer_dict.items())[:layer_num] if
                     not v]
    neuron_positions = [0 if not v else 1 for (layer_name, index), v in list(model_layer_dict.items())[:layer_num]]
    features = np.hstack((np.array(neuron_values).reshape(-1), np.array(neuron_positions).reshape(-1))).reshape(1, -1)
    benign_features.append(features)

#%%
np.save('adv_imageNet/VGG19/Imagenet_VGG19_Benign.npy', benign_features)

#%%
# benign_features = np.load('adv_cifar10/VGG19/VGG19_CIFAR10_benign_500_noposition.npy')
benign_features = np.array(benign_features)
benign_features = benign_features[:,0,:]

#%%
for i in range(10):
    path = 'adv_imagenet/vgg/adv_examples/FGSM/adv_x' + str(i) + '.npy'
    if i == 0:
        advs = np.load(path)[:20]
    else:
        advs = np.vstack((advs,np.load(path)[:20]))
# advs = np.load('adv_cifar10/vgg19/FGSM/adv.npy')
# advs = advs[:, 0, :, :, :]
#%%
layer_num = 4096
adv_features = []
num_sample = 200
for i in range(num_sample):
    x = advs[i]
    x = np.expand_dims(x, axis=0)
    model_layer_dict = init_position_tables(model)
    print(i)
    neuron_values = get_activation_values(x, model)
    get_position(x, model, model_layer_dict, threshold=0.4)
    ### total neuron = 4096
    not_activated = [(layer_name, index) for (layer_name, index), v in list(model_layer_dict.items())[:layer_num] if
                     not v]
    neuron_positions = [0 if not v else 1 for (layer_name, index), v in list(model_layer_dict.items())[:layer_num]]
    features = np.hstack((np.array(neuron_values).reshape(-1), np.array(neuron_positions).reshape(-1))).reshape(1, -1)
    adv_features.append(features)
#%%
adv_features = np.array(adv_features)
adv_features = adv_features[:, 0, :]
#%%
np.save('adv_cifar10/VGG19/cifar_PGD_vgg19.npy', adv_features)
#%%
for i in range(10):
    path = 'adv_imagenet/vgg/adv_examples/FGSM/adv_x' + str(i) + '.npy'
    if i == 0:
        detect_example = np.load(path)[20:40]
    else:
        detect_example = np.vstack((detect_example,np.load(path)[20:40]))
# detect_example = np.load('/data0/jinhaibo/DGAN/adv_cifar10/vgg19/FGSM/adv.npy')
#%%
# detect_example = detect_example[:,0,:,:,:]
detect_features = []
layer_num = 4096
num1 = 200
for i in range(num1):
    x = detect_example[i]
    x = np.expand_dims(x, axis=0)
    model_layer_dict = init_position_tables(model)
    print(i)
    neuron_values = get_activation_values(x, model)
    get_position(x, model, model_layer_dict, threshold=0.4)
    ### total neuron = 4096
    not_activated = [(layer_name, index) for (layer_name, index), v in list(model_layer_dict.items())[:layer_num] if
                     not v]
    neuron_positions = [0 if not v else 1 for (layer_name, index), v in list(model_layer_dict.items())[:layer_num]]
    features = np.hstack((np.array(neuron_values).reshape(-1), np.array(neuron_positions).reshape(-1))).reshape(1, -1)
    detect_features.append(features)
#%%
detect_features = np.array(detect_features)
detect_features = detect_features[:,0,:]

#%%
for i in range(8):
    path = 'adv_cifar10/Vgg16/clean/' + str(i) + '.npy'
    if i == 0:
        benign_features = np.load(path)
    else:
        benign_features = np.vstack((benign_features, np.load(path)))
benign_features = benign_features[: , 0, :]
#%%
# benign_features = np.load('adv_cifar10/VGG19/cifar_noise_vgg19.npy')
for i in range(8):
    path = 'adv_imagenet/mobile/PWA/' + str(i) + '.npy'
    if i == 0:
        adv_features = np.load(path)
    else:
        adv_features = np.vstack((adv_features, np.load(path)))
adv_features = adv_features[:, 0, :]

#%%
for i in range(8):
    path = 'adv_cifar10/Vgg16/Boundary/' + str(i) + '.npy'
    if i == 0:
        detect_features = np.load(path)
    else:
        detect_features = np.vstack((detect_features, np.load(path)))

detect_features = detect_features[:, 0, :]
# %%
num_benign = 400
num_sample = 200
num1 = 200
benign_features = np.array(benign_features).reshape(num_benign, -1)
adv_features = np.array(adv_features).reshape(num_sample, -1)
detect_features = np.array(detect_features).reshape(num1, -1)
benign_labels = np.array([0 for i in range(num_benign)]).reshape(num_benign, -1)
adv_labels = np.array([1 for i in range(num_sample)]).reshape(num_sample, -1)
detect_label = np.array([1 for i in range(num1)]).reshape(num1, -1)
# %%
sec_num = 200
x_to_train = np.vstack((benign_features[:sec_num], adv_features[:sec_num]))
y_to_train = np.vstack((benign_labels[:sec_num], adv_labels[:sec_num]))
x_to_test = np.vstack((benign_features[sec_num:], detect_features))
y_to_test = np.vstack((benign_labels[sec_num:], detect_label))
y_to_train_one = to_categorical(y_to_train, 2)
y_to_test_one = to_categorical(y_to_test, 2)


def meta_classify(input_shape):
    x = Dense(1028, name='train')(input_shape)
    # x = Dense(128, name='fc2', activation='relu')(x)
    x = Dense(2, name='predict', activation='softmax')(x)
    model = Model(input_shape, x)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


# %%
model_detect = meta_classify(input_shape=Input(shape=x_to_train.shape[1:]))
model_detect.fit(x_to_train, y_to_train_one, batch_size=5, epochs=20)
model_detect.save('adv_cifar10/vgg16/vgg19_FGSM_0.4.h5')
# model_detect = load_model('GTSRB/ResNet20_checker.h5')
# %%ACC score
loss, acc = model_detect.evaluate(x_to_test[sec_num:], y_to_test_one[sec_num:])
loss_benign, acc_benign = model_detect.evaluate(x_to_test[:sec_num], y_to_test_one[:sec_num])
print("Loss: %f; acc: %f; loss_beign: %f; acc_benign: %f" % (loss, acc, loss_benign, acc_benign))
# %% AUC score
detect_pred_test = model_detect.predict(x_to_test)
roc_value = roc_auc_score(y_to_test_one, detect_pred_test)
print('roc_value', roc_value)

