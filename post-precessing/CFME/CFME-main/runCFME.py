
import os
from mecfTracker import mecfTracker
import cv2
from get_image import get_image
import numpy as np
import sys
sys.path.append('./')

def calc_iou(bbox1, bbox2):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    bbox1[:,2:]+=bbox1[:,:2]
    bbox2[:,2:]+=bbox2[:,:2]
    xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)
 
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
 
    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))
 
    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)
    intersect = h * w
 
    union = area1 + np.squeeze(area2, axis=-1) - intersect
    return intersect / union

# set your video path
ROOT='/data/MOT/ICPR2024/track/'
vid=20
det_results='yolo-ship-nms'
DET_CONFIDENCE=0.39
data_path = '/data/MOT/ICPR2024/data/ICPR-1024/test/{}/img'.format('%03d'%vid)
det_path=os.path.join(ROOT,'dets',det_results,'{}.txt'.format('%03d'%vid))
output_path=os.path.join(ROOT,'CFME/CFME-main','output','{}.txt'.format('%03d'%vid))
if os.path.exists(output_path):
    os.remove(output_path)

dets=open(os.path.join(det_path)).readlines()
frame_dets={}
for det in dets:
    det=det.split(',')#1,-1,97.55,302.1,5.87,7.92,0.36,1,1
    if int(det[0]) not in frame_dets.keys():
        frame_dets[int(det[0])]=[]
    frame_dets[int(det[0])].append(det[:-1])

# dets=[det for det in dets if det.split(',')[0]=='1']
track_id=0
start_index=min(frame_dets.keys())
last_frame_dets=[]
track_list={}
for frame in range(min(frame_dets.keys()),max(frame_dets.keys())+1):
    last_frame_dets=[]
    if vid in [17,18,19,20]:
        img=cv2.imread(os.path.join(data_path,'{}.jpg'.format('%06d'%frame)))
    else:
        img=cv2.imread(os.path.join(data_path,'{}.png'.format('%03d'%frame)))
    # 
    print('%03d'%frame)
    for track_id,tracker in track_list.items():
        _, bbox = tracker.update(img)
        # bbox = list(map(int, map(np.round, bbox)))
        # bbox = list(map(float, map(np.round, bbox)))
        # bbox = list(np.array(bbox).round(2))
        if bbox[1]+bbox[3]<10:
            continue
        last_frame_dets.append([frame,track_id]+bbox)
    if frame in frame_dets.keys():
        dets=frame_dets[frame]
    else:
        dets=[]
    #计算当前帧检测和跟踪器预测的IOU，如果IOU>0.5，跳过
    if frame==start_index:#初始帧，为每个检测创建一个跟踪器
        for det in dets:
            # det=det.split(',')
            if float(det[6])<DET_CONFIDENCE:#TODO检测置信度
                continue
            # set init bounding box
            bbox = [float(det[2])-1,float(det[3])-1,float(det[4])+3,float(det[5])+3]
            cate=det[7]
            track_id+=1
            # cap = get_image(data_path)
            tracker = mecfTracker()           
            tracker.init(img, bbox)
            track_list[track_id]=tracker
            last_frame_dets.append([frame,track_id]+bbox)
    else:
        for det in dets:
            # det=det.split(',')
            bbox = [float(det[2])-1,float(det[3])-1,float(det[4])+3,float(det[5])+3]
            #与last_frame_dets计算IOU
            if last_frame_dets:
                iou_matrix = calc_iou(np.asarray([bbox]), np.array(last_frame_dets)[:,2:]) 
                
                indices = np.where(iou_matrix > 0.1)[1]
                if len(indices)!=0:
                    # index=indices.tolist()[0]
                    # last_frame_dets[index]=bbox
                    # track_list[index].update(img,bbox)
                    continue
            #如果没重合设置新跟踪器
            if float(det[6])>DET_CONFIDENCE:#TODO检测置信度
                # set init bounding box
                cate=det[7]
                # cap = get_image(data_path)
                #向前跟踪-用完弃用
                track_id+=1
                tracker = mecfTracker()           
                tracker.init(img, bbox)
                last_frame_dets.append([frame,track_id]+bbox)
                for fr in range(frame-1,0,-1):
                    fr_img=cv2.imread(os.path.join(data_path,'{}.jpg'.format('%06d'%fr)))
                    _, fr_bbox = tracker.update(fr_img)
                    # bbox=
                    if fr_bbox[1]+fr_bbox[3]<10:
                        continue
                    last_frame_dets.append([fr,track_id]+fr_bbox)

                #重新初始化一个跟踪器
                new_tracker = mecfTracker()           
                new_tracker.init(img, bbox)
                track_list[track_id]=new_tracker
                
            
            # cv2.rectangle(frame,(bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (255,255,255), 1)
            # cv2.imshow("video", frame)
            # cv2.waitKey(100)
    for det in last_frame_dets:
        frame_id=det[0]
        track_id=det[1]
        det=det[2:]
        with open(output_path,'a') as f:
            f.write(','.join([str(frame_id),str(track_id),'%.02f'%(det[0]),'%.02f'%(det[1]),'%.02f'%(det[2]),'%.02f'%(det[3]),'1',cate,'1\n']))
        
        
    
