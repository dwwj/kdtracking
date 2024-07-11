#可视化
import os
from pathlib import Path
import cv2
import numpy as np
import shutil
import xml.etree.ElementTree as ET # 读取xml
import os
from PIL import Image,ImageDraw,ImageFont



def visMOT(model,datapath,resultpath,isVisPic,isSavePic,isGenVID):
    '''
    datapath:数据集
    resultpath:某个模型的跟踪结果
    isVisPic:是否可视化图片
    isSavePic:是否保存图片
    isGenVID:是否生成视频
    '''
    
    videos=sorted(os.listdir(resultpath))
    for vid in videos:
        imgpath=os.path.join(datapath,vid.split('.')[0],'img')
        track_results=os.path.join(resultpath,vid)
        if not os.path.exists(track_results):
            print(vid+'无输出！')
            continue

       
        track={}

        if isGenVID:
            videopath=Path(os.path.join(VideoPath,model))
            videopath.mkdir(parents=True, exist_ok=True)
            # if vid.split() 
            # i=cv2.imread(os.path.join(imgpath,'000001.jpg'))
            # w=i.shape[1]
            # h=i.shape[0]
            w,h=1024,1024
            # videoWriter = cv2.VideoWriter(videopath+'/'+path+'.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (w, h))
            videoWriter = cv2.VideoWriter(os.path.join(videopath,vid+'.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), 15, (w,h))

        with open(track_results,'r') as f:
            lines=f.readlines()
        lines=[x.split(',') for x in lines]
       
        imglist=sorted(os.listdir(imgpath))
        for img in imglist:#遍历每一张图片
            img_id=int(img.split('.')[0])
            for l in lines:
                if int(l[0])==img_id:
                    x = float(l[2])
                    y = float(l[3])
                    w = float(l[4])
                    h = float(l[5])
                    if l[1] not in track.keys():
                        track[l[1]]=[tuple((np.random.rand(3) * 255).astype(int).tolist())]#随机生成颜色
                    track[l[1]].append((int(x), int(y),int(x+w),int(y+h)))
                    

            img=cv2.imread(os.path.join(imgpath,img))
            for key,value in track.items():
                color=value[0]
                b=value[-1]
                img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color, 2)
                for n in range(1,len(value)-1):
                    #中心点
                    #p1=(int((value[i][0] + value[i][2])/ 2), int((value[i][1] + value[i][3]) / 2))
                    #p2=(int((value[i+1][0] + value[i+1][2]) / 2), int((value[i+1][1] + value[i+1][3]) / 2))
                    #右下角
                    p1 = (int(value[n][2]), int(value[n][3]))
                    p2=(int(value[n+1][2]), int(value[n+1][3]))
                    img= cv2.line(img, p1, p2, color, 1)

            if isSavePic:
                savepic=Path(os.path.join(PicPath,model,vid))
                savepic.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(os.path.join(savepic,'%06d'%img_id+'.png'), img)
            if isVisPic:
                cv2.imshow('vis', img)
                cv2.waitKey(5)
            
            if isGenVID:
                # videoWriter.write(img[700:1080,292:660,:])#(292,700)(660,1080)
                videoWriter.write(img)

        if isVisPic:    
            cv2.destroyWindow('vis')
        if isGenVID:
            videoWriter.release()


def visYOLO():
    pass


if __name__ == "__main__":
    ROOT='/data/MOT/ICPR2024'
    DataPath=os.path.join(ROOT,'data')
    VideoPath=Path(os.path.join(ROOT,'Result','Vis','video'))
    PicPath=Path(os.path.join(ROOT,'Result','Vis','picture'))
    TrackPath=os.path.join(ROOT,'Result','Tracks')
    VideoPath.mkdir(parents=True, exist_ok=True)
    PicPath.mkdir(parents=True, exist_ok=True)

    # models=['DSFNet-pro-1-test']
    # 可视化MOT跟踪结果
    # for model in models:
    #     print(model)
    #     resultpath=os.path.join(TrackPath,model)
    imgpath='/data/MOT/ICPR2024/data/ICPR-1024/test/'
    trackresultpath='/data/MOT/ICPR2024/track/CFME/CFME-main/output/'
    visMOT('CFME',imgpath,trackresultpath,isVisPic=True,isSavePic=True,isGenVID=True)


        