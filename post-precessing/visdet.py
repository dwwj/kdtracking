#可视化数据集跟踪轨迹2024/5/7
import os
import cv2
import numpy as np
import shutil

det_confidence=0.1
ROOT='/data/MOT/ICPR2024/track/dets'
dataroot='/data/MOT/ICPR2024/data/ICPR-1024/test'
detroot=os.path.join(ROOT,'yolo-ship-nms')
resultroot=os.path.join(ROOT,'vis')
if os.path.exists(resultroot):
    shutil.rmtree(resultroot)
os.mkdir(resultroot)
for vid in sorted(os.listdir(detroot)):
    vid=int(vid.split('.')[0])
    datapath=os.path.join(dataroot,'{}/img'.format('%03d'%vid))
    detpath=os.path.join(detroot,'{}.txt'.format('%03d'%vid))
    resultpath=os.path.join(resultroot,'{}'.format('%03d'%vid))
    os.mkdir(resultpath)
    dets=open(detpath).readlines()
    frame_dets={}
    for det in dets:
        det=det.split(',')#1,-1,97.55,302.1,5.87,7.92,0.36,1,1
        if int(det[0]) not in frame_dets.keys():
            frame_dets[int(det[0])]=[]
        frame_dets[int(det[0])].append(det[:-1])
    
    for frame_id,det in frame_dets.items():
        imgpath=os.path.join(datapath,'{}.png'.format('%03d'%frame_id)) if vid not in [17,18,19,20] else os.path.join(datapath,'{}.jpg'.format('%06d'%frame_id))
        
        img=cv2.imread(imgpath)

        for d in det:#l:frame,id,x,y,w,h,1,1,1
            # id = int(l[1])
            x1 = int(float(d[2]))
            y1 = int(float(d[3]))
            x2 = int(float(d[4])+float(d[2]))
            y2 = int(float(d[5])+float(d[3]))
            score=str(d[6])
            if float(score)>det_confidence:
                img = cv2.rectangle(img, (x1, y1), (x2, y2), [0,255,0], 1)
                cv2.putText(img, score, (x1+1,y1+1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

               
        cv2.imwrite(os.path.join(resultpath,'{}.png'.format('%03d'%frame_id)), img)
        # cv2.imshow('{}-{}'.format(vid,frame_id), img)
        # cv2.waitKey(5)
        # cv2.destroyWindow('{}-{}'.format(vid,frame_id))
       