import os
path = "./datasets/VOCdevkit/VOC2007/JPEGImages"
filenames = os.listdir(path)
for filename in filenames:
  filename = path+"/"+filename
  python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_s.py -c YOLOX_outputs/yolox_voc_s/latest_ckpt.pth --path filename --conf 0.3 --nms 0.65 --tsize 640 --save_result --device gpu
