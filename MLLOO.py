# %%
import tensorflow as tf
import os
from keras.utils import to_categorical
import math
import time
import numpy as np
from keras.models import load_model
import sys
import os
import tarfile
import math
import zipfile
import keras.backend as K
import math
import pickle as pkl
from keras.models import Model
from scipy.stats import kurtosis, skew
from scipy.spatial.distance import pdist
import time
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.spatial.distance import pdist
from keras.datasets.cifar10 import load_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.compat.v1.disable_eager_execution()


# %%
def con(score):
    # score (n, d)
    score = score.reshape(len(score), -1)
    score_mean = np.mean(score, -1, keepdims=True)
    c_score = score - score_mean
    c_score = np.abs(c_score)
    return np.mean(c_score, axis=-1)


def mad(score):
    pd = []
    for i in range(len(score)):
        d = score[i]
        median = np.median(d)
        abs_dev = np.abs(d - median)
        med_abs_dev = np.median(abs_dev)
        pd.append(med_abs_dev)
    pd = np.array(pd)
    return pd


def med_pdist(score):
    pd = []
    for i in range(len(score)):
        d = score[i]
        k = np.median(pdist(d.reshape(-1, 1)))
        pd.append(k)
    pd = np.array(pd)
    return pd


def pd(score):
    pd = []
    for i in range(len(score)):
        d = score[i]
        k = np.mean(pdist(d.reshape(-1, 1)))
        pd.append(k)
    pd = np.array(pd)
    return pd


def neg_kurtosis(score):
    k = []
    for i in range(len(score)):
        di = score[i]
        ki = kurtosis(di, nan_policy='raise')
        k.append(ki)
    k = np.array(k)
    return -k


def quantile(score):
    # score (n, d)
    score = score.reshape(len(score), -1)
    score_75 = np.percentile(score, 75, -1)
    score_25 = np.percentile(score, 25, -1)
    score_qt = score_75 - score_25
    return score_qt


def calculate(score, stat_name):
    if stat_name == 'variance':
        results = np.var(score, axis=-1)
    elif stat_name == 'std':
        results = np.std(score, axis=-1)
    elif stat_name == 'pdist':
        results = pd(score)
    elif stat_name == 'con':
        results = con(score)
    elif stat_name == 'med_pdist':
        results = med_pdist(score)
    elif stat_name == 'kurtosis':
        results = neg_kurtosis(score)
    elif stat_name == 'skewness':
        results = -skew(score, axis=-1)
    elif stat_name == 'quantile':
        results = quantile(score)
    elif stat_name == 'mad':
        results = mad(score)
    print('results.shape', results.shape)
    return results


# %%
def evaluate_features(x, model, features):
    x = np.array(x)
    if len(x.shape) == 3:
        _x = np.expand_dims(x, 0)
    else:
        _x = x

    batch_size = 500
    num_iters = int(math.ceil(len(_x) * 1.0 / batch_size))
    generate_out = K.function([model.input], features)
    outs = []
    for i in range(num_iters):
        x_batch = _x[i * batch_size: (i + 1) * batch_size]
        # out = model.sess.run(features,
        #                      feed_dict={model.input_ph: x_batch})
        out = generate_out(x_batch)
        outs.append(out)

    num_layers = len(outs[0])
    outputs = []
    for l in range(num_layers):
        outputs.append(np.concatenate([outs[s][l] for s in range(len(outs))]))

    outputs = np.concatenate(outputs, axis=1)
    prob = outputs[:, -model.num_classes:]
    label = np.argmax(prob[-1])
    print('outputs', outputs.shape)
    print('prob[:, label]', np.expand_dims(prob[:, label], axis=1).shape)
    outputs = np.concatenate([outputs, np.expand_dims(prob[:, label], axis=1)], axis=1)

    return outputs


# %%

def collect_layers(model, interested_layers):
    outputs = [layer.output for layer in model.layers]

    outputs = [output for i, output in enumerate(outputs) if i in interested_layers]
    print(outputs)
    features = []
    for output in outputs:
        print(output)
        if len(output.get_shape()) == 4:
            features.append(
                tf.reduce_mean(output, axis=(1, 2))
            )
        else:
            features.append(output)
    return features


def loo_ml_instance(sample, reference, model, features):
    h, w, c = sample.shape
    sample = sample.reshape(-1)
    reference = reference.reshape(-1)

    data = []
    st = time.time()
    positions = np.ones((h * w * c + 1, h * w * c), dtype=np.bool)
    for i in range(h * w * c):
        positions[i, i] = False

    data = np.where(positions, sample, reference)

    data = data.reshape((-1, h, w, c))
    features_val = evaluate_features(data, model, features)  # (3072+1, 906+1)
    st1 = time.time()

    return features_val


# %%
def generate_ml_loo_features(reference, model, x, interested_layers):
    # print(args.attack)
    features = collect_layers(model, interested_layers)

    cat = {'original': 'ori', 'adv': 'adv', 'noisy': 'noisy'}
    dt = {'train': 'train', 'test': 'test'}
    stat_names = ['std', 'variance', 'con', 'kurtosis', 'skewness', 'quantile', 'mad']

    combined_features = {data_type: {} for data_type in ['test', 'train']}
    for data_type in ['test', 'train']:
        print('data_type', data_type)
        for category in ['original', 'adv']:
            print('category', category)
            all_features = []
            for i, sample in enumerate(x[data_type][category]):
                print('Generating ML-LOO for {}th sample...'.format(i))
                features_val = loo_ml_instance(sample, reference, model, features)

                # (3073, 907)
                print('features_val.shape', features_val.shape)
                features_val = np.transpose(features_val)[:, :-1]
                print('features_val.shape', features_val.shape)
                # (906, 3073)
                single_feature = []
                for stat_name in stat_names:
                    print('stat_name', stat_name)
                    single_feature.append(calculate(features_val, stat_name))

                single_feature = np.array(single_feature)
                print('single_feature', single_feature.shape)
                # (k, 906)
                all_features.append(single_feature)
            print('all_features', np.array(all_features).shape)
            combined_features[data_type][category] = np.array(all_features)

            np.save('/data0/jinhaibo/lixiaohao/adv_cifar10/VGG16/{}_{}.npy'.format(
                dt[data_type],
                cat[category]),
                combined_features[data_type][category])

    return combined_features


# %%
# (x_train, y_train), (x_test, y_test) = load_data()
# x_train = x_train / 255.0
# x_test = x_test / 255.0
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)
# x_train1 = np.load('/data0/jinhaibo/DGAN/animals_10_datasets/vgg/train/img_data.npy')
# x_train1 = x_train1/255.0
# y_train1 = np.load('/data0/jinhaibo/DGAN/animals_10_datasets/vgg/train/img_data_label.npy')
# x_train = []
# y_train = []
# for i in range(10):
#     if i == 0:
#         x_train = x_train1[y_train1 == i][:40]
#         y_train = y_train1[y_train1 == i][:40]
#     else:
#         x_train = np.vstack((x_train, x_train1[y_train1 == i][:40]))
#         y_train = np.hstack((y_train, y_train1[y_train1 == i][:40]))
# X_train = x_train[:200]
# X_test = x_train[200: 400]
# advs_train = []
# advs_test = []
# for i in range(10):
#     path = '/data0/jinhaibo/DGAN/adv_imagenet/mobile/FGSM/adv_x' + str(i)+'.npy'
#     adv = np.load(path)
#     advs_train.extend(adv[:20])
#     advs_test.extend(adv[20:50])
# advs = np.vstack((advs_train, advs_test))

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
advs = np.load('/data0/jinhaibo/DGAN/Inverse_Peturbation/Voiceprint_nip/DeepSpeaker/Boundary/advs.npy')
X_train = x_train[:200]
X_test = x_train[200: 400]
X_train_adv = advs[:200]
X_test_adv = advs[200: 400]

#%%
x = {
    'train': {
        'original': X_train,
        'adv': X_train_adv,
    },
    'test': {
        'original': X_test,
        'adv': X_test_adv,
    },
}
model = load_model('/data0/jinhaibo/DGAN/Inverse_Peturbation/Voiceprint_nip/models/DeepSpeaker_all.h5')
model.summary()
# %%
cat = {'original': 'ori', 'adv': 'adv', 'noisy': 'noisy'}
dt = {'train': 'train', 'test': 'test'}
interested_layers = [54, 55, 56]
print('extracting layers ', interested_layers)
reference = - np.zeros(x_train.shape[1:])
combined_features = generate_ml_loo_features(reference, model, x, interested_layers)


# %%
def train_and_evaluate(detections, attack, fpr_upper=1.0):
    auc_dict = {}
    tpr1 = {}
    tpr5 = {}
    tpr10 = {}

    for det in detections:
        # Load data
        x, y = load_data(args, attack, det)
        x_train, y_train = x['train'], y['train']
        x_test, y_test = x['test'], y['test']
        x_train = x_train.reshape(len(x_train), -1)
        x_test = x_test.reshape(len(x_test), -1)
        # Train
        lr = LogisticRegressionCV(n_jobs=-1).fit(x_train, y_train)
        # Predict
        pred = lr.predict_proba(x_test)[:, 1]
        pred = lr.predict_proba(x_test)[:, 1]
        # Evaluate.
        fpr, tpr, thresholds = roc_curve(y_test, pred)

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        auc_dict[det] = auc(fpr, tpr)
        tpr1[det] = tpr[find_nearest(fpr, 0.01)]
        tpr5[det] = tpr[find_nearest(fpr, 0.05)]
        tpr10[det] = tpr[find_nearest(fpr, 0.10)]
        plt.plot(
            fpr, tpr,
            label="{0} (AUC: {1:0.3f})".format(labels[det], auc(fpr, tpr)),
            color=color_dict[det],
            linestyle=linestyles[det],
            linewidth=4)

    plt.xlim([0.0, fpr_upper])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=32)
    plt.ylabel('True Positive Rate', fontsize=32)
    plt.title(
        '{} ({}, {})'.format(labels_attack[attack], labels_data[args.dataset_name], labels_model[args.model_name]),
        fontsize=32)
    plt.legend(loc="lower right", fontsize=22)
    plt.show()
    figure_name = '{}/figs/mad_transfer_roc_{}_{}_{}.pdf'.format(args.data_model, args.data_sample, attack, attack)
    plt.savefig(figure_name)

    return auc_dict, tpr1, tpr5, tpr10
