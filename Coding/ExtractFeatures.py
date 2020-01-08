#!/usr/bin/env python

'''
    Extract HOG and HOF Descriptors
    ================================
'''


import numpy as np
import cv2
import pickle
import Features
import os
import datetime
from shutil import copyfile
import errno


def descriptors(root,max,width,height):

    Traj=open(root+"/Trayectories.pkl",'rb')
    Trayectories=pickle.load(Traj)

    print(root)
    print(datetime.datetime.now().time())

    HOG_Descriptors=[]
    HOF_Descriptors=[]
    ntraj=0;

    for trajectory in Trayectories:
        HOG_Descriptors.append([])
        HOF_Descriptors.append([])
        #print(ntraj)

        for i in range(len(trajectory)):

            nframe=ntraj*15

            descriptor_HOG=[]
            descriptor_HOF=[]
            position=0

            while(position<15 and nframe<max-2):

                if (trajectory[i][position][1]-15>0 and trajectory[i][position][1]+17<height and trajectory[i][position][0]-15>0 and trajectory[i][position][0]+17<width):

                    frame=cv2.imread(root+"/frame "+str(nframe)+".jpg")
                    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    flow=np.load(root+"/flow "+str(nframe+1)+".npy")

                    if (position==0):

                        HOG=np.zeros((4,8),np.float32)
                        HOF=np.zeros((4,8),np.float32)

                        window=frame[int(trajectory[i][position][1]-15):int(trajectory[i][position][1]+17),int(trajectory[i][position][0]-15):int(trajectory[i][position][0]+17)]
                        window_flow=flow[int(trajectory[i][position][1]-15):int(trajectory[i][position][1]+17),int(trajectory[i][position][0]-15):int(trajectory[i][position][0]+17)]

                        #print(str(trajectory[i][position][1]) + " "+ str(trajectory[i][position][0]))

                        HOG=HOG+Features.HOG_32(window)
                        HOF=HOF+Features.HOF_32(window_flow)



                    elif ((position+1)%5==0):
                        #print(position)
                        window=frame[int(trajectory[i][position][1]-15):int(trajectory[i][position][1]+17),int(trajectory[i][position][0]-15):int(trajectory[i][position][0]+17)]
                        window_flow=flow[int(trajectory[i][position][1]-15):int(trajectory[i][position][1]+17),int(trajectory[i][position][0]-15):int(trajectory[i][position][0]+17)]

                        HOG=HOG+Features.HOG_32(window)
                        HOF=HOF+Features.HOF_32(window_flow)

                        descriptor_HOG.extend(HOG[0])
                        descriptor_HOG.extend(HOG[1])
                        descriptor_HOG.extend(HOG[2])
                        descriptor_HOG.extend(HOG[3])

                        descriptor_HOF.extend(HOF[0])
                        descriptor_HOF.extend(HOF[1])
                        descriptor_HOF.extend(HOF[2])
                        descriptor_HOF.extend(HOF[3])

                        HOG=np.zeros((4,8),np.float32)
                        HOF=np.zeros((4,8),np.float32)

                    else:
                        #print(position)
                        window=frame[int(trajectory[i][position][1]-15):int(trajectory[i][position][1]+17),int(trajectory[i][position][0]-15):int(trajectory[i][position][0]+17)]
                        window_flow=flow[int(trajectory[i][position][1]-15):int(trajectory[i][position][1]+17),int(trajectory[i][position][0]-15):int(trajectory[i][position][0]+17)]


                        HOG=HOG+Features.HOG_32(window)
                        HOF=HOF+Features.HOF_32(window_flow)

                else:
                    if (position==0):
                        HOG=np.zeros((4,8),np.float32)
                        HOF=np.zeros((4,8),np.float32)

                nframe=nframe+1;
                position=position+1;

            #print(i)
            #print(len(descriptor_HOG))
            #print(len(descriptor_HOF))

            HOG_Descriptors[ntraj].append(descriptor_HOG)
            HOF_Descriptors[ntraj].append(descriptor_HOF)
        ntraj=ntraj+1;



    HOG_file=open(root+"/HOG.pkl",'wb')
    pickle.dump(HOG_Descriptors,HOG_file)


    HOF_file=open(root+"/HOF.pkl",'wb')
    pickle.dump(HOF_Descriptors,HOF_file)

def main():

    print(__doc__)
    import iDT as Traj
    import sys

    filename=sys.argv[1]
    root = os.path.splitext(filename)[0]
    try:
        os.makedirs(root)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
            pass

    if (sys.argv[2]=='DIS'):
        tr=root+"/Trayectories_DIS.pkl"
        if os.path.isfile(tr):
            a=0
        else:
            Traj.App(filename).run(root,'DIS')
        copyfile(tr,root+"/Trayectories.pkl")

    elif (sys.argv[2]=='Deep'):
        tr=root+"/Trayectories_DIS.pkl"
        if os.path.isfile(tr):
            a=0
        else:
            Traj.App(filename).run(root,'DIS')
        copyfile(tr,root+"/Trayectories.pkl")
    else:
        print ("ERROR: ----------Incorrect Argument")
        exit()

    max=len(os.listdir(root))/2
    HOG=root+"/HOG.pkl"
    HOF=root+"/HOF.pkl"
    if os.path.isfile(HOG) and os.path.isfile(HOF):
        a=1
    else:
        print('Extracting HOG  and HOG Features...')
        cam = cv2.VideoCapture(filename)
        ret, frame = cam.read()
        height, width = frame.shape[:2]
        descriptors(root,max,width,height)
        cam.release()
        os.remove(root+"/Trayectories.pkl")

if __name__ == '__main__':
    main()
