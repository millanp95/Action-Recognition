#!/usr/bin/env python


import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn import svm
from common import anorm2, draw_str

import cv2
import pickle
import os
import sys
import datetime
import shutil
import time
import random
import errno
import itertools
import matplotlib.pyplot as plt

import trajectories_DIS as Traj
import ExtractFeatures


GMM_params = dict(n_components=int(sys.argv[1]),
                  covariance_type='diag',
                  tol=0.001, reg_covar=1e-06,
                  max_iter=100, n_init=1,
                  init_params='kmeans',
                  weights_init=None,
                  means_init=None,
                  precisions_init=None,
                  random_state=None,
                  warm_start=False,
                  verbose=1,
                  verbose_interval=10)


SVM_params= dict(penalty='l2',
                 loss='squared_hinge',
                 dual=False,
                 tol=0.0001,
                 C=1.0,
                 multi_class='ovr',
                 fit_intercept=True,
                 intercept_scaling=1,
                 class_weight=None,
                 verbose=1,
                 random_state=None,
                 max_iter=1000)


def Extract_Trajectories(folder,videolist):

    for videofile in videolist:
        filename=os.path.join(folder,videofile)
        root = os.path.splitext(filename)[0]
        try:
            os.makedirs(root)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        Traj.App(filename).run(root)

    cv2.destroyAllWindows()

def Fisher_Encoding(X,GMM):
    #Fisher Vector Encoding Algorithm. (Input X(T,96), Output: Fisher Encoding)

    FV=[] #Empty Fisher Vector, we will fill it through the cicles.

    probabilities=GMM.predict_proba(X)
    n,m = X.shape[:2]

    #print (probabilities.shape)

    for k in range (GMM_params["n_components"]):
        #0-2 order Statistics Initialization.
        S_0=np.array([0.0]);
        S_1=np.zeros(192);
        S_2=np.zeros(192);

        for T in range(n):
            S_0=S_0+[probabilities[T,k]];
            S_1=S_1+probabilities[T,k]*X[T,:];
            S_2=S_2+probabilities[T,k]*(X[T,:]**2); #Point-wise product. Check !!!!!!

        FV.extend(S_0)
        FV.extend(S_1)
        FV.extend(S_2)


    FV=normalize(np.array(FV).reshape(1,-1),norm='l2')

    return FV

def Extract_Features(folder,videolist):

    for videofile in videolist:
        filename=os.path.join(folder,videofile)
        root = os.path.splitext(filename)[0]
        max=len(os.listdir(root))/2
        #ExtractFeatures.descriptors(root,max)

        HOG=root+"/HOG.pkl"
        HOF=root+"/HOF.pkl"

        if os.path.isfile(HOG) and os.path.isfile(HOF):
        	continue
        else:
            ExtractFeatures.descriptors(root,max)

def show_classification(root,label):

    video=root+".mp4"
    cap = cv2.VideoCapture(video)
    cv2.namedWindow('current video',1);
    nframe = 0;

    if (cap.isOpened()):
        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret==True:
                draw_str(frame, (20, 20), "class: "+label)
                cv2.imshow('current video',frame)
                cv2.waitKey(10)

            else:
                break
        cap.release()
    else:
        print("ERROR: Cannot open file")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.1f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if normalize:
        plt.savefig('Normalized.eps', format='eps', dpi=1000)
    else:
        plt.savefig('Confs.eps', format='eps', dpi=1000)

def SVM_video_classify(root,Machine,GMM):

    if os.path.isfile(root+"/HOG.pkl") and os.path.isfile(root+"/HOG.pkl"):

        HOG_file=open(root+"/HOG.pkl",'rb')
        HOG=pickle.load(HOG_file)

        HOF_file=open(root+"/HOF.pkl",'rb')
        HOF=pickle.load(HOF_file)

        Features=[]


        for i in range(len(HOG)):
            x=[]
            for j in range(len(HOG[i])):
                if (len(HOG[i][j])==96 and len(HOF[i][j])==96):
                    sample=[]
                    sample.append(HOG[i][j])
                    sample.append(HOF[i][j])
                    sample=normalize(sample,norm='l2').reshape(1,-1)
                    sample=sample.tolist()
                    x.append(sample)
                else:
                    continue

            if (len(x)>1):
                X=np.array(x)
                X=X.reshape(X.shape[0],-1)
                n,m = X.shape[:2]

                #print(X.shape)

                if n > 0:
                    FisherVector=Fisher_Encoding(X,GMM)
                else:
                    continue

                #print(FisherVector.shape)
                Features.append(FisherVector)
            else:
                continue

        n_traj=len(Features)
        #print(n_traj)
        if n_traj>0:
            Features=np.asarray(Features)
            #print(Features.shape)
            Features=Features.reshape(n_traj,-1)

            final_prediction=[]

            for i in range(Features.shape[0]):
                labels=[]
                distances=np.zeros(len(Machine))
                for j in range(len(Machine)):
                    feat=Features[i,:]
                    feat=feat.reshape(1,-1)
                    #print(Machine[j].predict(feat))
                    labels.append(Machine[j].predict(feat)[0])
                    #print(np.max(Machine[j].decision_function(feat)))
                    distances[j]=np.max(Machine[j].decision_function(feat))
                #print(labels)
                #print(distances)
                final_prediction.append(labels[np.argmax(distances)])
                #print("Final prediction",final_prediction)

            final_prediction=np.asarray(final_prediction)

            #print (root,final_prediction)
            pick= np.sum(final_prediction == 'pick')
            drop= np.sum(final_prediction == 'drop')
            handshake= np.sum(final_prediction == 'handshake')
            nothing= np.sum(final_prediction == 'nothing')
            a='pick'
            #print(final_prediction==a)

            highest=max(pick,drop,handshake,nothing)

            if (highest==pick):
                label="pick"
            elif (highest== drop):
                label="drop"
            elif (highest== handshake):
                label="handshake"
            else:
                label="nothing"
        else:
            label="nothing"


        show_classification(root,label)

        return label

def SVM_classify(GMM,Machine,folder,list):

    random.shuffle(list)

    classes=["pick","drop","handshake","nothing"]

    confussion_mtrx=np.zeros([4,4])

    for videofile in list:

        label=videofile.split("_")[0]
        filename=os.path.join(folder,videofile)
        root = os.path.splitext(filename)[0]

        if root != "./Labeled/.DS_Store":
            #print(root)
            label=classes.index(label)
            classification=SVM_video_classify(root,Machine,GMM)

            if classification in classes:
                classification=classes.index(classification)

            print(label,classification)
            confussion_mtrx[label,classification]+=1


    confussion_file=open("confussion_mtrx"+sys.argv[1]+"_"+".pkl",'wb')
    pickle.dump(confussion_mtrx,confussion_file)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confussion_mtrx, classes=classes,
                          title='Confusion matrix, without normalization')


    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confussion_mtrx, classes=classes, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

    print(confussion_mtrx)
    print("Error")
    print(1-np.trace(confussion_mtrx)/np.sum(confussion_mtrx))

def main():

    Features_Folder="./Labeled"
    text_file = open(sys.argv[2]+".txt", "r")
    videolist = text_file.read().split('\n')
    videolist=videolist[:len(videolist)-1]

    GMM_file=open("GMM_"+sys.argv[1]+".pkl",'rb')
    print("loading GMM ...")
    GMM=pickle.load(GMM_file)

    SVM_file=open("Machine_"+sys.argv[3]+".pkl",'rb')
    print("loading Machine...")
    Machine=pickle.load(SVM_file)

    #Extract_Trajectories(Features_Folder,videolist)
    #Extract_Features(Features_Folder,videolist)
    SVM_classify(GMM,Machine,Features_Folder,videolist)


if __name__ == '__main__':
    main()
