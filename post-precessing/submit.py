import os
import shutil
import codecs
import xml.etree.ElementTree as ET
import cv2

def submit_det(det_path,img_path,xml_root):
    if os.path.exists(xml_root):
        shutil.rmtree(xml_root)
    os.mkdir(xml_root)

    video_path='/data/MOT/ICPR2024/data/ICPR-1024/test/'
    video_len={}
    videos=sorted(os.listdir(video_path))
    for vid in videos:
        vid_path=os.path.join(video_path,vid,'img')
        video_len[int(vid)]=len(os.listdir(vid_path))

    for vid in range(1,21):
        if not os.path.exists(os.path.join(det_path,'{}.txt'.format('%03d'%vid))):
            continue

        dets=open(os.path.join(det_path,'{}.txt'.format('%03d'%vid))).readlines()

        #读取并复制图像
        img=cv2.imread(os.path.join(img_path,'MOT17-{}-FRCNN/img1/000001.jpg'.format('%02d'%vid)))   
        height,width,depth = img.shape

        for frame in range(1,video_len[vid]+1):
            xml_path=os.path.join(xml_root,'{}_{}.xml'.format('%03d'%vid,'%06d'%frame))
            #生成xml头文件-描述图片信息
            with codecs.open(xml_path, 'a', 'utf-8') as xml:
                xml.write('<annotation>\n')
                xml.write('\t<folder>' + 'VOC2007' + '</folder>\n')
                xml.write('\t<filename>' + '{}_{}.xml'.format('%03d'%vid,'%06d'%frame) + '</filename>\n')
                xml.write('\t<path>' + xml_root + '</path>\n')
                xml.write('\t<source>\n')
                xml.write('\t\t<database>The UAV autolanding</database>\n')
                xml.write('\t</source>\n')
                xml.write('\t<size>\n')
                xml.write('\t\t<width>' + str(width) + '</width>\n')
                xml.write('\t\t<height>' + str(height) + '</height>\n')
                xml.write('\t\t<depth>' + str(depth) + '</depth>\n')
                xml.write('\t</size>\n')
                xml.write('\t\t<segmented>0</segmented>\n')
                xml.write('</annotation>')

            frame_dets=[det for det in dets if int(det.split(',')[0])==frame]
            xml_path=os.path.join(xml_root,'{}_{}.xml'.format('%03d'%vid,'%06d'%frame))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for det in frame_dets:
                det=det.split(',')#frame_id,track_id(-1),x,y,w,h,score,class,visibility
                bbox=[det[2],det[3],'%.02f'%(float(det[4])+float(det[2])),'%.02f'%(float(det[5])+float(det[3]))]
                score=det[6]
                if det[7]=='1':
                    cate='car'
                if det[7]=='3':
                    cate='ship'
                object = ET.SubElement(root, "object")  

                name = ET.SubElement(object,'name')
                name.text = cate

                pose = ET.SubElement(object,'pose')
                pose.text = '0'

                truncated = ET.SubElement(object,'truncated')
                truncated.text = '1'

                difficult = ET.SubElement(object,'difficult')
                difficult.text = '0'
                
                bndbox = ET.SubElement(object,'bndbox')
                xmin = ET.SubElement(bndbox,'xmin')
                xmin.text = bbox[0]
                ymin = ET.SubElement(bndbox,'ymin')
                ymin.text = bbox[1]
                xmax = ET.SubElement(bndbox,'xmax')
                xmax.text = bbox[2]
                ymax = ET.SubElement(bndbox,'ymax')
                ymax.text = bbox[3]
                
                extra = ET.SubElement(object,'extra')
                extra.text = score
            tree.write(xml_path)

def submit_track(track_root,out_path):
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)
    for cate in ['car','ship']:
    # for cate in ['ship']:
        track_path=os.path.join(track_root,cate)
        videos=sorted(os.listdir(track_path))
        for vid in videos:
            # dets=open(os.path.join(det_path,'{}.txt'.format('%03d'%vid))).readlines()
            dets=open(os.path.join(track_path,vid)).readlines()
            with open(os.path.join(out_path,vid),'a') as f:
                for i in range(len(dets)):
                    det=dets[i]
                    det=det.split(',')
                    if cate=='car':
                        det[7]='1'
                    if cate=='ship':
                        det[7]='3'
                    det[8]='1\n'
                    det.pop()
                    f.write(','.join(det))

ROOT='/data/MOT/ICPR2024/track/'  
img_path='/data/MOT/ICPR2024/data/MOT17/viso-test/test/'
submit_root=os.path.join(ROOT,'submit/')
det_path=os.path.join(ROOT,'dets','FFCA-car-0.4')
out_det_path=os.path.join(submit_root,'det')
track_path=os.path.join(ROOT,'track','DSFNet')
out_track_path=os.path.join(submit_root,'track')
# submit_det(det_path,img_path,out_det_path)
submit_track(track_path,out_track_path)