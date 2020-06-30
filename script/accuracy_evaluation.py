# Evaluation accuracy
import PIL
import os
import numpy as np
import  matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score,recall_score
import sys
indicator = raw_input("Attention U-net(1) or U-net(0):")
if indicator == '0':
    name = ['preds_test_total_aug_up_U_net.tif','preds_test_total_aug_down_U_net.tif','preds_test_total_aug_left_U_net.tif','preds_test_total_aug_right_U_net.tif']
else:
    name = ['preds_test_total_aug_up_attention_U_net.tif','preds_test_total_aug_down_attention_U_net.tif','preds_test_total_aug_left_attention_U_net.tif','preds_test_total_aug_right_attention_U_net.tif']
ref = np.load('../data/reference.npy')
mask = np.load('../data/mask.npy')
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
import copy
from scipy.io import loadmat
for i in range(len(name)):
    if name[i] == '':
        continue
    if name[i] == 'NHD.mat' or name[i] == 'Geonet.tif':
        x = loadmat(name[i])
        I = x['NHD']
        I[I>0] = 1
        [lrr,lcr] = np.where(mask == 1)
    else:    
        I = copy.copy(np.asarray(PIL.Image.open('./result/'+name[i])))
        if i == 0:
            half = ref.shape[0]//2
            reft = ref[half:]
            It = I[half:]
            maskt = mask[half:]
            [lr,lc] = np.where(maskt == 1)
            lr += half
            [lrt,lct] = np.where(It == 1)
            lrt += half
            [lrr,lcr] = np.where(reft == 1)
            lrr += half        
        elif i == 1:
            half = ref.shape[1]//2
            reft = ref[:,half:]
            It = I[:,half:]
            maskt = mask[:,half:]
            [lr,lc] = np.where(maskt == 1)
            lc += half
            [lrt,lct] = np.where(It == 1)
            lct += half        
            [lrr,lcr] = np.where(reft == 1)
            lcr += half 
        elif i == 2:
            half = ref.shape[0]//2
            reft = ref[:half,:]
            It = I[:half,:]
            maskt = mask[:half]
            [lr,lc] = np.where(maskt == 1) 
            [lrt,lct] = np.where(It == 1)
            [lrr,lcr] = np.where(reft == 1)
        else:
            half = ref.shape[1]//2
            reft = ref[:,:half]
            It = I[:,:half]
            maskt = mask[:,:half]
            [lr,lc] = np.where(maskt == 1)
            [lrt,lct] = np.where(It == 1)
            [lrr,lcr] = np.where(reft == 1)
    #if i < 4:
#    for p1,p2 in zip(lrt,lct):    
#        if I[p1,p2] == 1 and ref[p1,p2] == 0 and sum(sum(ref[(p1-2):(p1+2),(p2-2):(p2+2)])) > 0:
#            ref[p1,p2] = 1
    It = copy.copy(I)
    for d1,d2 in zip(lrr,lcr):    
        if I[d1,d2] == 0 and ref[d1,d2] == 1 and sum(sum(I[(d1-3):(d1+3),(d2-3):(d2+3)])) > 0:
            It[d1,d2] = 1
    I = It
    print('#######################'+name[i]+'#######################')
    #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    #axes[0].imshow(I*255,cmap='gray',vmin=0, vmax=255)
    #axes[1].imshow(ref*255,cmap='gray',vmin=0, vmax=255)
    #plt.imshow(ref*255,cmap='gray',vmin=0, vmax=255)
    #plt.show()    
    groundtruthlist = ref[lr,lc]
    predictionlist = I[lr,lc]
    cm = confusion_matrix(groundtruthlist, predictionlist)
    print(cm)
    #plot_confusion_matrix(cm,classes=["Non-streams","Streams"])
    print('F1 score of stream: '+str(f1_score(groundtruthlist, predictionlist,pos_label=1)))
    print('F1 score of nonstream: '+str(f1_score(groundtruthlist, predictionlist,pos_label=0)))
    print('Precision of stream: '+str(precision_score(groundtruthlist, predictionlist,pos_label=1)))
    print('Precision of nonstream: '+str(precision_score(groundtruthlist, predictionlist,pos_label=0)))
    print('Recall of stream: '+str(recall_score(groundtruthlist, predictionlist,pos_label=1)))
    print('Recall of nonstream: '+str(recall_score(groundtruthlist, predictionlist,pos_label=0)))    
    print('#######################end#######################')
