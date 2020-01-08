#!/usr/bin/env python

'''
     Extract Trajectories
    ====================
    '''

from __future__ import print_function

import numpy as np
import cv2
from common import anorm2, draw_str
import os
import shutil
import errno
import pickle
import sys


feature_params = dict( maxCorners = 1000,
                      qualityLevel = 0.01,
                      minDistance = 5,
                      blockSize = 5 )

class App:
    def __init__(self, video_src):
        self.track_len = 15
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.Block_idx = -1

    def run(self,root,flow_tch):
        if (flow_tch=='DIS'):
            inst = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM)

        elif (flow_tch=='Deep'):
            inst = cv2.optflow.createOptFlow_DeepFlow()

        else:
            print ("ERROR: ----------Incorrect Argument")
            exit()

        while True:
            ret, frame = self.cam.read()

            if (ret==True):

                height, width = frame.shape[:2]

                flow=np.zeros((height,width,2),np.float32)
                flow_inv=np.zeros((height,width,2),np.float32)


                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #vis = frame.copy()

                if (self.frame_idx % self.track_len == 0):
                    self.tracks.append([])
                    self.Block_idx += 1
                    mask = np.zeros_like(frame_gray)
                    p = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks[self.Block_idx].append([(x, y)])

                    if (self.frame_idx>0):
                        inst.calc(self.prev_gray,frame_gray,flow)
                        flow = cv2.medianBlur(flow,3)

                else:
                    img0, img1 = self.prev_gray, frame_gray
                    inst.calc(img0,img1,flow)
                    inst.calc(img1,img0,flow_inv)

                    p0 = np.float32([tr[-1] for tr in self.tracks[self.Block_idx]]).reshape(-1, 1, 2)
                    p1 = np.zeros_like(p0)
                    p0r = np.zeros_like(p0)

                    cont=0;
                    for pt in p0:
                        x,y = pt.ravel()
                        x=int(np.rint(x))
                        y=int(np.rint(y))

                        if (y+flow[y,x,1]>height-1):
                            y=height-1
                        else:
                            y=y+flow[y,x,1];

                        x=int(np.rint(x))
                        y=int(np.rint(y))

                        if (x+flow[y,x,0]>width-1):
                            x=width-1
                        else:
                            x=x+flow[y,x,0];

                        new_point=[x,y];
                        p1[cont]=new_point
                        cont +=1

                    cont=0;
                    for pt in p1:
                        x,y = pt.ravel()
                        x=int(np.rint(x))
                        y=int(np.rint(y))

                        if (y+flow_inv[y,x,1]>height-1):
                            y=height-1
                        else:
                            y=y+flow_inv[y,x,1];

                        x=int(np.rint(x))
                        y=int(np.rint(y))

                        if (x+flow_inv[y,x,0]>width-1):
                            x=width-1
                        else:
                            x=x+flow_inv[y,x,0];

                        new_point=[x,y];
                        p0r[cont]=new_point
                        cont +=1

                    d = abs(p0-p0r).reshape(-1, 2).max(-1)
                    good = d < 1.25
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks[self.Block_idx], p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        new_tracks.append(tr)
                        #cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

                    self.tracks[self.Block_idx] = new_tracks
                    #cv2.polylines(vis, [np.int32(tr) for tr in self.tracks[self.Block_idx]], False, (0, 255, 0))

                    #draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks[self.Block_idx]))

                    #cv2.imshow('DIS-Flow_track', vis)
                    #cv2.waitKey(10)
                self.frame_idx += 1
                self.prev_gray = frame_gray

            else:
                break

        if (flow_tch=='DIS'):
            file=open(root+"/Trayectories_DIS.pkl",'wb')
            pickle.dump(self.tracks,file)

        elif (flow_tch=='Deep'):
            file=open(root+"/Trayectories_Deep.pkl",'wb')
            pickle.dump(self.tracks,file)


        #print(self.tracks)

        #file=open(root+"/Trayectories.pkl",'rb')
        #self.tracks=pickle.load(file)

        for i in range(len(self.tracks)):
            j=0;
            while j < len(self.tracks[i]):
                x=np.array(self.tracks[i][j])
                #print('..............point........')
                #print(x)
                distance= np.sum(np.linalg.norm(np.diff(x,axis=0),axis=1))
                #print('.........Distance...........')
                #print(distance)

                if distance<15:
                    self.tracks[i].pop(j)
                else:
                    j+=1;

        #print('--------------new_tracks--------')
        #print(self.tracks)

        self.cam.set(cv2.CAP_PROP_POS_FRAMES,0)
        frame_idx=0
        Block_idx=-1

        while True:
            ret, frame = self.cam.read()

            if (ret==True):
                if (frame_idx % self.track_len == 0):
                    frame_idx = 0
                    Block_idx+=1

                #print('Indices')
                #print(Block_idx,frame_idx)
                for tr in self.tracks[Block_idx]:
                    cv2.circle(frame, tr[frame_idx], 2, (0, 255, 0), -1)
                    cv2.polylines(frame, [np.int32(tr[:frame_idx]) for tr in self.tracks[Block_idx]], False, (0, 255, 0))
                    draw_str(frame, (20, 20), 'track count: %d' % len(self.tracks[Block_idx]))

                cv2.imshow('DIS-Flow_track', frame)
                cv2.waitKey(10)

                frame_idx +=1

            else:
                break


def main():

    print(__doc__)

    #try:
    #    videofile = sys.argv[1]
    #except:
    #    videofile = 0

    #folder="../Labeled"
    #filename=os.path.join(folder,videofile)
    filename=sys.argv[1]
    flow_tch=sys.argv[2]
    root = os.path.splitext(filename)[0]
    try:
        os.makedirs(root)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
            pass
    App(filename).run(root,flow_tch)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
