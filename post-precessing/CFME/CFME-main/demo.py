import os
from mecfTracker import mecfTracker
import cv2
from get_image import get_image
import numpy as np
import sys
sys.path.append('./')
# set your video path
ROOT='/data/MOT/ICPR2024/track/'
vid=17
start_index=1
DET_CONFIDENCE=0.4
model='yolo-ship-nms'
data_path = '/data/MOT/ICPR2024/data/ICPR-1024/test/{}/img'.format('%03d'%vid)
det_path=os.path.join(ROOT,'dets',model,'{}.txt'.format('%03d'%vid))
output_path=os.path.join(ROOT,'CFME/CFME-main','output','{}.txt'.format('%03d'%vid))
if os.path.exists(output_path):
    os.remove(output_path)


dets=open(os.path.join(det_path)).readlines()
dets=[det for det in dets if int(det.split(',')[0])==start_index]

track_id=0
for det in dets:
    det=det.split(',')
    if float(det[6])<DET_CONFIDENCE:
        continue
    # set init bounding box
    bbox = [float(det[2]),float(det[3]),float(det[4])-1,float(det[5])-1]
    cate=det[7]
    track_id+=1
    cap = get_image(data_path)
    tracker = mecfTracker()
    index = start_index

    for frame in cap:
        if index == start_index:
            tracker.init(frame, bbox)            
        else:
            _, bbox = tracker.update(frame)
            # bbox_draw = list(map(int, map(np.round, bbox)))
            # bbox = list(map(float, bbox))
            
            # cv2.rectangle(frame,(bbox_draw[0],bbox_draw[1]), (bbox_draw[0]+bbox_draw[2],bbox_draw[1]+bbox_draw[3]), (255,255,255), 1)
            # cv2.imshow("video", frame)
            # cv2.waitKey(100)
        with open(output_path,'a') as f:
            f.write(','.join([str(index),str(track_id),"%.02f"%(bbox[0]),"%.02f"%(bbox[1]),"%.02f"%(bbox[2]),"%.02f"%(bbox[3]),'1',cate,'1\n']))
        index += 1
