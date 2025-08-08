# %%
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

#%%
aa = np.load('features4demo/CIFAR10/VGG19/FGSM/Adv/features_0.npy')

# %%
for i in range(80):
    path_Benign = 'features4demo/CIFAR10/VGG19/FGSM/Benign/features_' + str(i) + '.npy'
    path_adv = 'features4demo/CIFAR10/VGG19/FGSM/Adv/features_' + str(i) + '.npy'
    if i == 0:
        Benign_feature = np.load(path_Benign)
        adv_feature = np.load(path_adv)
    else:
        Benign_feature = np.vstack((Benign_feature, np.load(path_Benign)))
        adv_feature = np.vstack((adv_feature, np.load(path_adv)))

Benign_feature = Benign_feature[:,:4096]
Benign_label = np.array([0 for i in range(len(Benign_feature))])
Benign_label = to_categorical(Benign_label, 2)
adv_feature = adv_feature[:,:4096]
adv_label = np.array([1 for i in range(len(adv_feature))])
adv_label = to_categorical(adv_label, 2)

# %%
num_train = 200
x_train = np.vstack((Benign_feature[:num_train], adv_feature[:num_train]))
y_train = np.vstack((Benign_label[:num_train], adv_label[:num_train]))


# %%
def meta_classify(input_shape):
    x = Dense(1028, name='train')(input_shape)
    # x = Dense(128, name='fc2', activation='relu')(x)
    x = Dense(2, name='predict', activation='softmax')(x)
    model = Model(input_shape, x)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


# %%
model_detect = meta_classify(input_shape=Input(shape=x_train.shape[1:]))
model_detect.fit(x_train, y_train, batch_size=5, epochs=10)
#%%
model_detect.save('features4demo/model_without.h5')
# %%
predict = model_detect.predict(Benign_feature[num_train:])
predict = np.argmax(predict, axis=1)
index = []
for i in range(len(predict)):
    if predict[i] == 0:
        index.append(i + 200)

# %%
np.save('features4demo/CIFAR10/VGG19/index.npy', index)

# %%
names_layer = 'train'


def get_activation_values(input_data, model):
    layer_names = [layer.name for layer in model.layers if names_layer in layer.name]
    get_value = [[] for j in range(len(layer_names))]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = intermediate_layer_output
        for num_neuron in range(scaled.shape[-1]):
            get_value[i].append(np.mean(scaled[..., num_neuron]))
    return get_value

class_feature = []
for i in index:
    print(i)
    feature_1 = get_activation_values(Benign_feature[i:i+1], model_detect)
    class_feature.append(feature_1)
#%%
class_feature = np.array(class_feature)
class_feature = class_feature[:,0, :]

#%%
np.save('features4demo/Benign_classifeir_po.npy', class_feature)
