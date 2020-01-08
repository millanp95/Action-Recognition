#!/sur/bin/env python

"""
    ====================
    4 classes SVM-Train
    ====================

    *Nothing.
    *HandShake.
    *Pick.
    *Drop.

"""

import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MaxAbsScaler
from sklearn import svm

import cv2
import pickle
import os
import datetime
import errno
import sys
import time

import trajectories_DIS as Traj
import ExtractFeatures

GMM_params = dict(n_components=int(sys.argv[1]),
                  covariance_type='diag',
                  tol=0.001, reg_covar=1e-06,
                  max_iter=200, n_init=5,
                  init_params='kmeans',
                  weights_init=None,
                  means_init=None,
                  precisions_init=None,
                  random_state=None,
                  warm_start=False,
                  verbose=0,
                  verbose_interval=10)

SVM_params= dict(penalty='l2',
                 loss='squared_hinge',
                 dual=False,
                 tol=0.0001,
                 C=2.0,
                 multi_class='ovr',
                 fit_intercept=True,
                 intercept_scaling=1,
                 class_weight=None,
                 verbose=0,
                 random_state=None,
                 max_iter=1000)

def classes(label):
    return{
        "drop": 1,
        "pick": 2,
        "handshake": 3,
        "nothing": 4,
    }[label]

def RemoveFiles(root):
    files = os.listdir(root)

    for file in files:
        if file.endswith(".jpg") or file.endswith(".npy"):
            os.remove(os.path.join(root,file))

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

def Extract_Features(folder,videolist):

    for videofile in videolist:

        start_time = time.time()

        filename=os.path.join(folder,videofile)
        root = os.path.splitext(filename)[0]
        try:
            os.makedirs(root)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass

        tr=root+"/Trayectories.pkl"
        if os.path.isfile(tr):
        	continue
        else:
        	Traj.App(filename).run(root)

        max=len(os.listdir(root))/2
        HOG=root+"/HOG.pkl"
        HOF=root+"/HOF.pkl"
        if os.path.isfile(HOG) and os.path.isfile(HOF):
        	continue
        else:
            print('Extracting HOG  and HOG Features...')
            cam = cv2.VideoCapture(filename)
            ret, frame = cam.read()
            height, width = frame.shape[:2]
            ExtractFeatures.descriptors(root,max,width,height)
            cam.release()
        print("--- %s seconds ---" % (time.time() - start_time))
        RemoveFiles(root)

def FitGMM(Features_Folder, videolist):

    data_HOG=[]
    data_HOF=[]

    new_data=[]
    labels=[]

    print ("Selecting the videos ...")
    for videofile in videolist:

        filename=os.path.join(Features_Folder,videofile)
        root = os.path.splitext(filename)[0]

        HOG_file=open(root+"/HOG.pkl",'rb')
        HOG=pickle.load(HOG_file)

        HOF_file=open(root+"/HOF.pkl",'rb')
        HOF=pickle.load(HOF_file)

        #print(label)

        for trajectory in HOG:
        	data_HOG.extend(trajectory)

        for trajectory in HOF:
                data_HOF.extend(trajectory)

    print ("Selecting good samples ...")
    #Aqui ya estamos normalizando HOG y HOF
    for i in range(len(data_HOG)):
        if (len(data_HOG[i])==96 and len(data_HOF[i])==96):
            sample=[]
            sample.append(data_HOG[i])
            sample.append(data_HOF[i])
            sample=normalize(sample,norm='l2').reshape(1,-1)
            sample=sample.tolist()
            new_data.append(sample)
        else:
            continue

    new_data=np.array(new_data)
    print(new_data.shape)

    print("reshaping...")
    new_data=new_data.reshape(new_data.shape[0],-1)
    print(new_data.shape)

    #print ("Saving good samples...")
    #data_file=open("4_classes_data.pkl",'wb')
    #pickle.dump(new_data,data_file)

    '''
    data_file=open("4_classes_data.pkl",'rb')
    print("loading data...")
    new_data=pickle.load(data_file)
    print(new_data.shape)
    '''

    GMM = GaussianMixture(**GMM_params)
    print("Fitting Gaussian Mixture Model")
    GMM.fit(new_data)
    new_data=[]

    print( "Gaussian Fitting Complete")
    print( "Saving GMM ...")

    return GMM

def Fisher_Vector(X,GMM):
    #Fisher Vector Encoding Algorithm. (Input X(T,192), Output: Fisher Encoding)

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

def Fisher_Encoding(Features_Folder,videolist,GMM):

    print("Encoding Fisher Vectors...")

    for videofile in videolist:

        filename=os.path.join(Features_Folder,videofile)
        root = os.path.splitext(filename)[0]

        HOG_file=open(root+"/HOG.pkl",'rb')
        HOG=pickle.load(HOG_file)

        HOF_file=open(root+"/HOF.pkl",'rb')
        HOF=pickle.load(HOF_file)

        #print(filename)

        cont = 0
        #print(len(HOG))


        for i in range(len(HOG)):
            x=[]
            for j in range(len(HOG[i])):
                if (len(HOG[i][j])==96 and len(HOF[i][j])==96):
                    sample=[]
                    sample.append(HOG[i][j])
                    sample.append(HOF[i][j])
                    sample=np.nan_to_num(sample)
                    sample=normalize(sample,norm='l2').reshape(1,-1)
                    sample=sample.tolist()
                    x.append(sample)
                else:
                    continue

            if (len(x)>1):
                X=np.array(x)
                X=X.reshape(X.shape[0],-1)
                n,m = X.shape[:2]

                if n > 0:
                    FisherVector=Fisher_Vector(X,GMM)
                    Fisher_file=open(root+"/Fisher_"+str(cont)+".pkl",'wb')
                    pickle.dump(FisherVector,Fisher_file)
                    cont+=1
                else:
                    continue
            else:
                continue

        cont_file=open(root+"/cont.pkl",'wb')
        pickle.dump(cont,cont_file)

def Fisher_Load(Features_Folder,videolist):

    print("Loading Fisher Vectors...")
    Final_Features=[];
    Final_Labels=[];

    for videofile in videolist:
        label=videofile.split("_")[0]

        filename=os.path.join(Features_Folder,videofile)
        root = os.path.splitext(filename)[0]

        root = os.path.splitext(filename)[0]

        cont_file=open(root+"/cont.pkl",'rb')
        cont=pickle.load(cont_file)

        for i in range(cont):
            Fisher_file=open(root+"/Fisher_"+str(i)+".pkl",'rb')
            FisherVector=pickle.load(Fisher_file)
            Final_Features.append(FisherVector)
            Final_Labels.append(label)


    Final_Features=np.asarray(Final_Features)
    Final_Features=Final_Features.reshape(Final_Features.shape[0],-1)
    Final_Labels=np.asarray(Final_Labels)

    return Final_Features,Final_Labels

def Boostrapping_Train(Features,Labels,penalty,N):

    machine=[]

    for i in range(N):
        SVM_params["C"]=penalty[i]
        SVM=svm.LinearSVC(**SVM_params)
        n=int(Features.shape[0]*0.8)
        index=np.random.choice(Features.shape[0],n, replace=True)
        Final_Features=Features[index,:]
        Final_Labels=Labels[index]
        SVM.fit(Final_Features,Final_Labels)
        machine.append(SVM)

    SVM_file=open("Machine_"+str(N)+".pkl",'wb')
    print("Saving Smodel...")
    pickle.dump(machine,SVM_file)

def main():

    Features_Folder="./Labeled"

    text_file = open("Test.txt", "r")
    videolist = text_file.read().split('\n')
    videolist=videolist[:len(videolist)-1]

    Extract_Features(Features_Folder,videolist)

    text_file = open("Train.txt", "r")
    videolist = text_file.read().split('\n')
    videolist=videolist[:len(videolist)-1]

    Extract_Features(Features_Folder,videolist)

    GMM=FitGMM(Features_Folder,videolist)

    #GMM_file=open("GMM_"+sys.argv[1]+".pkl",'rb')
    #print("loading GMM ...")
    #GMM=pickle.load(GMM_file)

    Fisher_Encoding(Features_Folder,videolist,GMM)

    Features,Labels=Fisher_Load(Features_Folder,videolist)
    Boostrapping_Train(Features,Labels,[0.5,1,2,5],4)


    #SVM = svm.LinearSVC(**SVM_params)
    #SVM.fit(Features,Labels)

    #pickle.dump(SVM,SVM_file)


if __name__ == '__main__':
    main()
