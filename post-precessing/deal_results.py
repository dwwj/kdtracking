#将模型生成的检测结果处理为mot15格式
#frame_id,track_id(-1),x,y,w,h,score,class,visibility
import os
import shutil
import json
import torch
import torchvision.ops as ops
#处理yolox结果
# split=[1]


def get_video_split(video_path):
    count=1
    split={}
    videos=sorted(os.listdir(video_path))
    for vid in videos:
        
        vid_path=os.path.join(video_path,vid,'img')
        # print(vid,len(os.listdir(vid_path)))
        split[vid]=[count,count+len(os.listdir(vid_path))-1]
        count+=len(os.listdir(vid_path))
    return split

def deal_yolo(results_path,det_path,det_confidence=0):
    global video_split

    # for cate in ['car','ship']:
    for cate in ['ship']:
        output_path=os.path.join(det_path,'yolo-{}'.format(cate))
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)
        dets=open(os.path.join(results_path,'yolox_{}_5.txt'.format(cate))).readlines()
        # dets=[x.split() for x in dets]#img_id,scores,boxes[x1,y1,x2,y2]

        
        for det in dets:
            det=det.split(' ')
            img_id=int(det[0])
            score=det[1]
            bbox=det[2:]
            if float(score)<det_confidence:
                continue
            # bbox[2:]-=bbox[:2]
            if cate=='car':
                cate='1'
            if cate=='ship':
                cate='3'
            for k,v in video_split.items():
                if img_id>=v[0] and img_id<=v[1]:
                    # if cate=='3' and float(score)<0.3:
                    #     continue
                    if cate=='3' and k not in ['017','020']:
                        break
                    with open(os.path.join(output_path,'{}.txt'.format(k)),'a') as fy:
                        if k=='020':
                            if ((float(bbox[2])-float(bbox[0]))<30 or (float(bbox[3])-float(bbox[1]))<10) or (float(bbox[1])<10 and (float(bbox[3])-float(bbox[1]))<25):
                                pass
                            else:
                                if (float(bbox[2])-float(bbox[0]))<35 or (float(bbox[3])-float(bbox[1]))<15:
                                    if (float(bbox[2])-float(bbox[0]))<35 and (float(bbox[3])-float(bbox[1]))<15:
                                        fy.write(','.join([str(img_id-v[0]+1),'-1',bbox[0],bbox[1],'35.0','15.0',score,cate,'1\n']))
                                    elif (float(bbox[2])-float(bbox[0]))<35:
                                        fy.write(','.join([str(img_id-v[0]+1),'-1',bbox[0],bbox[1],'35.0','%.02f'%(float(bbox[3])-float(bbox[1])),score,cate,'1\n']))
                                    elif (float(bbox[3])-float(bbox[1]))<15:
                                        fy.write(','.join([str(img_id-v[0]+1),'-1',bbox[0],bbox[1],'%.02f'%(float(bbox[2])-float(bbox[0])),'15.0',score,cate,'1\n']))
                       
                                else:
                                    fy.write(','.join([str(img_id-v[0]+1),'-1',bbox[0],bbox[1],'%.02f'%(float(bbox[2])-float(bbox[0])),'%.02f'%(float(bbox[3])-float(bbox[1])),score,cate,'1\n']))
                        if k=='017':
                            if (float(bbox[3])-float(bbox[1]))<10 or float(bbox[1])>500:
                                pass
                            
                            else:
                                # if (float(bbox[2])-float(bbox[0]))>30:
                                #     gap=(float(bbox[3])-float(bbox[1]))-30
                                #     fy.write(','.join([str(img_id-v[0]+1),'-1','%.02f'%(float(bbox[0])-gap),bbox[1],'30.0','%.02f'%(float(bbox[3])-float(bbox[1])),score,cate,'1\n']))
                                # else:
                                fy.write(','.join([str(img_id-v[0]+1),'-1',bbox[0],bbox[1],'%.02f'%(float(bbox[2])-float(bbox[0])),'%.02f'%(float(bbox[3])-float(bbox[1])),score,cate,'1\n']))
                        
                    break
#分数据集置信度阈值
def deal_DSFNet2(results_path,det_path,VIDEOS_CONFIDENCE):
    global video_split
    
    for cate in ['car','ship']:
        output_path=os.path.join(det_path,'DSFNet-{}-{}'.format(cate,'deal'))
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)
        with open(os.path.join(results_path,'DSFNet_{}.json'.format(cate)),'r', encoding='UTF-8') as f:
            dets = json.load(f)#{'image_id': 68, 'category_id': 1, 'bbox': [113.61, 0.95, 5.18, 5.1], 'score': 0.42}

        
        for det in dets:
            img_id=det['image_id']
            if cate=='car':
                cate='1'
            if cate=='ship':
                cate='3'
            bbox=det['bbox']
            score=det['score']
            # if score>det_confidence:
            
            if cate == '3':
                if img_id<=300 and score>0.5:
                    with open(os.path.join(output_path,'017.txt'),'a') as fd:
                        fd.write(','.join([str(img_id),'-1',str(bbox[0]),str(bbox[1]),str(bbox[2]),str(bbox[3]),str(score),cate,'1\n']))
                elif img_id<=600 and score>0.5:
                    with open(os.path.join(output_path,'020.txt'),'a') as fd:
                        fd.write(','.join([str(img_id-300),'-1',str(bbox[0]),str(bbox[1]),str(bbox[2]),str(bbox[3]),str(score),cate,'1\n']))
                
            else:
                for k,v in video_split.items():
                    if img_id>=v[0] and img_id<=v[1]:
                        if score>VIDEOS_CONFIDENCE[k]:
                            with open(os.path.join(output_path,'{}.txt'.format(k)),'a') as fd:
                                fd.write(','.join([str(img_id-v[0]+1),'-1',str(bbox[0]),str(bbox[1]),str(bbox[2]),str(bbox[3]),str(score),cate,'1\n']))
                        break
#统一置信度
def deal_DSFNet(results_path,det_path,det_confidence=0):
    global video_split
    
    for cate in ['car','ship']:
        output_path=os.path.join(det_path,'DSFNet-{}-{}'.format(cate,det_confidence))
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)
        with open(os.path.join(results_path,'DSFNet_{}.json'.format(cate)),'r', encoding='UTF-8') as f:
            dets = json.load(f)#{'image_id': 68, 'category_id': 1, 'bbox': [113.61, 0.95, 5.18, 5.1], 'score': 0.42}
        
        for det in dets:
            img_id=det['image_id']
            if cate=='car':
                cate='1'
            if cate=='ship':
                cate='3'
            bbox=det['bbox']
            score=det['score']
            if score>det_confidence:
                if cate == '3':
                    if img_id<=300:
                        with open(os.path.join(output_path,'017.txt'),'a') as fd:
                            fd.write(','.join([str(img_id),'-1',str(bbox[0]),str(bbox[1]),str(bbox[2]),str(bbox[3]),str(score),cate,'1\n']))
                    elif img_id<=600:
                        with open(os.path.join(output_path,'020.txt'),'a') as fd:
                            fd.write(','.join([str(img_id-300),'-1',str(bbox[0]),str(bbox[1]),str(bbox[2]),str(bbox[3]),str(score),cate,'1\n']))
                    
                else:
                    for k,v in video_split.items():
                        if img_id>=v[0] and img_id<=v[1]:
                            with open(os.path.join(output_path,'{}.txt'.format(k)),'a') as fd:
                                fd.write(','.join([str(img_id-v[0]+1),'-1',str(bbox[0]),str(bbox[1]),str(bbox[2]),str(bbox[3]),str(score),cate,'1\n']))
                            break


def deal_CT(results_path,det_path,det_confidence=0):
    global video_split
    
    # for cate in ['car','ship']:
    cate='ship'
    output_path=os.path.join(det_path,'CT-{}-{}'.format(cate,det_confidence))
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    with open(os.path.join(results_path,'CT-{}.json'.format(cate)),'r', encoding='UTF-8') as f:
        dets = json.load(f)#{'image_id': 68, 'category_id': 1, 'bbox': [113.61, 0.95, 5.18, 5.1], 'score': 0.42}
    
    for frame,det in dets.items():
        if det==[]:
            continue
        for d in det:
            img_id=int(frame)
            cate='3'#指定ship
            bbox=d['bbox']
            score=d['score']
            if score>det_confidence:
                for k,v in video_split.items():
                    if img_id>=v[0] and img_id<=v[1]:
                        with open(os.path.join(output_path,'{}.txt'.format(k)),'a') as fd:
                            fd.write(','.join([str(img_id-v[0]+1),'-1',str(bbox[0]),str(bbox[1]),'%.02f'%(float(bbox[2])-float(bbox[0])),'%.02f'%(float(bbox[3])-float(bbox[1])),str(score),cate,'1\n']))
                        break

def deal_FFCA(results_path,det_path,det_confidence=0):
    global video_split
    cate='car'
    output_path=os.path.join(det_path,'FFCA-{}-{}'.format(cate,det_confidence))
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    with open(os.path.join(results_path,'FFCA_{}.json'.format(cate)),'r', encoding='UTF-8') as f:
        dets = json.load(f)#{'image_id': 68, 'category_id': 1, 'bbox': [113.61, 0.95, 5.18, 5.1], 'score': 0.42}
  
    for det in dets:
        img_id=det['image_id']
        cate='1'
        bbox=det['bbox']
        score=det['score']
       
        for k,v in video_split.items():
            if img_id>=v[0] and img_id<=v[1]:
                # if score>VIDEOS_CONFIDENCE[k]:
                if score>det_confidence:
                    with open(os.path.join(output_path,'{}.txt'.format(k)),'a') as fd:
                        fd.write(','.join([str(img_id-v[0]+1),'-1',str(bbox[0]),str(bbox[1]),str(bbox[2]),str(bbox[3]),str(score),cate,'1\n']))
                break


def combine_det(yolo_det_path,dsfnet_det_path,results_path,conf_thres=0.1,iou_thres=0.3):
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.mkdir(results_path)
    for i in range(1,21):
        yolo_det=open(os.path.join(yolo_det_path,'{}.txt'.format('%03d'%i))).readlines()
        dsf_det=open(os.path.join(dsfnet_det_path,'{}.txt'.format('%03d'%i))).readlines()
        output_path=os.path.join(results_path,'{}.txt'.format('%03d'%i))
        if os.path.exists(output_path):
            os.remove(output_path)
        all_dets=yolo_det+dsf_det
        all_dets=[x.split(',') for x in all_dets]
        all_dets.sort(key=lambda x:int(x[0]))
        frame_dets={}
        for det in all_dets:
            if int(det[0]) not in frame_dets:
                frame_dets[int(det[0])]=[]
            frame_dets[int(det[0])].append([int(det[0]),float(det[1]),float(det[2]),float(det[3]),float(det[4]),int(det[5]),float(det[6])])
        with open(output_path,'a') as f:
            for frame,det in frame_dets.items():
                det.sort(key=lambda x:float(x[6]),reverse=True)
                #nms去重
                # boxes: Tensor, 预测框
                # scores: Tensor, 预测置信度
                # iou_threshold: float, IOU阈值
                det=torch.Tensor(det)
                bboxes=det[:,1:5]
                scores=det[:,5]
                i=ops.nms(bboxes,scores,iou_thres)
                new_det=det[i]
                for d in new_det:
                    # if float(d[6])>conf_thres:
                    f.write(','.join([str(int(d[0])),'%.02f'%(d[1]),'%.02f'%(d[2]),'%.02f'%(d[3]),'%.02f'%(d[4]),str(int(d[5])),'%.02f'%(d[6]),'\n']))
        # pass

def nms(det_path,results_path,iou_thres=0.5,conf_thres=0.3):
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.mkdir(results_path)
    videos=sorted(os.listdir(det_path))
    for vid in videos:
        i=int(vid.split('.')[0])
        dets=open(os.path.join(det_path,'{}.txt'.format('%03d'%i))).readlines()
        output_path=os.path.join(results_path,'{}.txt'.format('%03d'%i))
        # if os.path.exists(output_path):
        #     os.remove(output_path)
        
        dets=[x.split(',') for x in dets]
        dets.sort(key=lambda x:int(x[0]))
        frame_dets={}
        for det in dets:
            if int(det[0]) not in frame_dets:
                frame_dets[int(det[0])]=[]
            ##frame_id,track_id(-1),x,y,w,h,score,class,visibility
            frame_dets[int(det[0])].append([int(det[0]),float(det[2]),float(det[3]),float(det[4]),float(det[5]),float(det[6]),int(det[7])])
            
        with open(output_path,'a') as f:
            for frame,det in frame_dets.items():
                det.sort(key=lambda x:float(x[6]),reverse=True)
                #nms去重
                # boxes: Tensor, 预测框
                # scores: Tensor, 预测置信度
                # iou_threshold: float, IOU阈值
                det=torch.Tensor(det)
                bboxes=det[:,1:5].clone()
                bboxes[:,2:]+=bboxes[:,:2]
                scores=det[:,5]
                i=ops.nms(bboxes,scores,iou_thres)
                new_det=det[i]
                for d in new_det:
                    if float(d[5])>conf_thres:
                    ##frame_id,track_id(-1),x,y,w,h,score,class,visibility
                        f.write(','.join([str(int(d[0])),'-1','%.02f'%(d[1]),'%.02f'%(d[2]),'%.02f'%(d[3]),'%.02f'%(d[4]),'%.02f'%(d[5]),str(int(d[6])),'1\n']))
        # pass



VIDEOS_CONFIDENCE={'001':0.5,
                   '002':0.4,
                   '003':0.3,
                   '004':0.6,
                   '005':0.6,
                   '006':0.6,
                   '007':0.4,
                   '008':0.6,
                   '009':0.4,
                   '010':0.3,
                   '011':0.3,
                   '012':0.3,
                   '013':0.3,
                   '014':0.2,
                   '015':0.4,
                   '016':0.4,
                   '017':0.4,
                   '018':0.2,
                   '019':0.3,
                   '020':0.3}

video_path='/data/MOT/ICPR2024/data/ICPR-1024/test/'
video_split=get_video_split(video_path)

ROOT='/data/MOT/ICPR2024/track/'
results_path=os.path.join(ROOT,'dets/results')
det_path=os.path.join(ROOT,'dets/')
# deal_DSFNet(results_path,det_path,det_confidence=0.1)
# deal_CT(results_path,det_path,det_confidence=0.1)
# deal_DSFNet2(results_path,det_path,VIDEOS_CONFIDENCE)#为各个数据集设置专门置信度
deal_yolo(results_path,det_path,det_confidence=0.1)
nms(os.path.join(det_path,'yolo-ship'),os.path.join(det_path,'yolo-ship-nms'),iou_thres=0.2)
# deal_FFCA(results_path,det_path,det_confidence=0.5)

results_path=os.path.join(ROOT,'dets/combine')
# combine_det(det_path,results_path)