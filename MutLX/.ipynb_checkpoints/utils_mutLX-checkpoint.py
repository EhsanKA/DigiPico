"""Auxiliary methods used in MutLX

This module contains functions used in MutLX to build the models, pre-process the data, and plot the results.

   - `iter_loadtxt` is used for a faster line by line reading of large csv files.
   - `prep_typebased` reads and preprocesses data.
   - `build_model` contains the code to build main models used in this version.
   - `mutLX_plots` is used for plotting the outputs
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
from scipy import stats
import tensorflow as tf
from tensorflow import keras
import seaborn as sns

def iter_loadtxt(filename, usecols=None, delimiter=',', skiprows=0, dtype=np.float32):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                if usecols is not None:
                    line = [line[i] for i in usecols]
                for item in line:
                    yield item
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data

def prep_typebased(path_full, cols):
    print("Start Loading Data!")

    dataset = iter_loadtxt(path_full, usecols=cols, dtype='S20')
    names = iter_loadtxt(path_full, usecols=range(2), dtype='S20')

    print("Loading Data Done!")
    print("Start Preparing Data!")

    snp_unq_ind = [index for index, data_type in enumerate(dataset[:, 0]) if data_type == b'SNP-Unq']
    snp_hm_ind = [index for index, data_type in enumerate(dataset[:, 0]) if data_type == b'SNP-Hm']
    snp_ht_h_ind = [index for index, data_type in enumerate(dataset[:, 0]) if data_type == b'SNP-Ht-H']
    snp_ht_ind = [index for index, data_type in enumerate(dataset[:, 0]) if data_type == b'SNP-Ht']
    snp_sm_ind = [index for index, data_type in enumerate(dataset[:, 0]) if data_type == b'SNP-Somatic']

    neg_ind = snp_unq_ind
    pos_ind = snp_ht_h_ind + snp_ht_ind
    test_ind = pos_ind + neg_ind
    pos_ind = np.sort(pos_ind)
    test_ind = np.sort(test_ind)
    rest_pos_ind = snp_sm_ind + snp_hm_ind
    hpos_ind = snp_ht_h_ind + snp_hm_ind

    # Replacing the names with labels
    dataset[pos_ind, 0] = 1
    dataset[neg_ind, 0] = 0
    dataset[rest_pos_ind, 0] = 1

    dataset = np.float64(dataset)

    print("Data Preparation Done!")
    return dataset, test_ind, neg_ind, pos_ind, hpos_ind, names

def build_model(input_dim, output_dim, type, weights_path=''):
    tf.random.set_seed(7)
    
    if type == 'ml-binary':
        model = keras.Sequential([
            keras.layers.Dense(input_dim, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dense(int(input_dim/2), activation='relu'),
            keras.layers.Dense(output_dim, activation='sigmoid')
        ])
    elif type == 'ml-binary-dropout':
        model = keras.Sequential([
            keras.layers.Dense(input_dim, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.8),
            keras.layers.Dense(int(input_dim/2), activation='relu'),
            keras.layers.Dropout(0.7),
            keras.layers.Dense(output_dim, activation='sigmoid')
        ])
    else:
        raise ValueError("Invalid model type")

    if weights_path:
        model.load_weights(weights_path)

    opt = keras.optimizers.Adam()
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['binary_accuracy'])
    
    return model

def mutLX_plots(final, neg_ind, hpos_ind, minScore, auc_cf, tpr_cf, out):
    neg_set = final[neg_ind]
    pos_set = final[hpos_ind] 
    neg_set = neg_set[neg_set[:,0] > minScore]
    pos_set = pos_set[pos_set[:,0] > minScore]
    thr = []
    tpr = []
    utd = []
    for t in np.arange(0.25, -0.01, -0.01):
        thr.append(t)
        utd.append(len(neg_set[neg_set[:,1] <= t, 1]))
        tpr.append(len(pos_set[pos_set[:,1] <= t, 1]) / len(pos_set[:,1]))
    fpr = np.array(utd[:]) / utd[0]
    roc_auc = auc(fpr, tpr)

    cf = 0
    if roc_auc > auc_cf:
        i = 0
        old_thr = 0
        while cf == 0:
            if 1 - 4*thr[i] > tpr[i]:
                cf = (thr[i] + old_thr) / 2
            else:
                old_thr = thr[i]
            i += 1
    else:
        i = len(tpr) - 1
        old_thr = 0
        while cf == 0:
            if tpr[i] >= tpr_cf:
                cf = (thr[i] + old_thr) / 2
            else:
                old_thr = thr[i]
            i -= 1    

    plt.rcParams['axes.linewidth'] = 3

    data = np.column_stack((final, np.repeat("SNP", len(final[:,0]))))
    data[neg_ind, 2] = "UTDs"
    data[hpos_ind, 2] = "Germline SNPs"
    SNPs = ["UTDs", "Germline SNPs"]
    
    plt.figure(figsize=(10, 6))
    for SNP in SNPs:
        subset = data[data[:,2] == SNP]
        sns.kdeplot(subset[:,0].astype(float), shade=True, linewidth=3, clip=(0,1), label=SNP)
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    plt.ylim([-0.1, ymax])
    plt.xlim([-0.02, 1.02])
    plt.xlabel('Probability score')
    plt.ylabel('Density')
    plt.plot([minScore, minScore], [0, ymax+2], 'r--', linewidth=2)
    plt.legend(loc='upper right')
    plt.savefig(out + "_Probability_Score.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(utd, tpr, label=f'ROC curve (area = {roc_auc:.2f})', linewidth=3)
    plt.xlim([0.0, np.amax(utd)])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Number of passed UTDs (False Positives)')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right", prop={'size': 18})
    ax2 = plt.gca().twinx()
    ax2.plot(utd, thr, markeredgecolor='r', linestyle='dashed', color='r', linewidth=3)
    ax2.set_ylabel('Drop-out variance threshold', color='r')
    ax2.set_ylim([0.25, 0])
    ax2.set_xlim([0.0, np.amax(utd)])
    plt.savefig(out + "_ROC.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(final[neg_ind,1], final[neg_ind,0], c='grey', s=20, alpha=0.3)
    plt.xlim([-0.01, 0.21])
    plt.ylim([-0.01, 1.02])
    plt.plot([-0.01, 0.22], [minScore, minScore], 'r--', linewidth=2)
    plt.plot([cf, cf], [-0.2, 1.1], 'r--', linewidth=2)
    plt.xlabel('Uncertainty Score')
    plt.ylabel('Probability Score')
    plt.savefig(out + "_UTDs.png")
    plt.close()

    np.random.shuffle(final[hpos_ind])
    plt.figure(figsize=(10, 6))
    plt.scatter(final[hpos_ind[:1000],1], final[hpos_ind[:1000],0], c='b', s=20, alpha=0.3)
    plt.xlim([-0.01, 0.21])
    plt.ylim([-0.01, 1.02])
    plt.plot([-0.01, 0.22], [minScore, minScore], 'r--', linewidth=2)
    plt.plot([cf, cf], [-0.2, 1.1], 'r--', linewidth=2)
    plt.xlabel('Uncertainty Score')
    plt.ylabel('Probability Score')
    plt.savefig(out + "_Germline.png")
    plt.close()

    return cf
