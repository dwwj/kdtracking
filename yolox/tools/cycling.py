import os
'''str="python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_s.py -c YOLOX_outputs/yolox_voc_s/latest_ckpt.pth --path ./datasets/VOCdevkit/VOC2007/JPEGImages/"
int i=0
for i in range(0,10):
    j = "%06d" % (i + 1)
    lines = str+j+".jpg --conf 0.3 --nms 0.65 --tsize 640 --save_result --device gpu"   #python命令 + B.py + 参数：IC.txt'
    print(lines)
    os.system(lines)'''

if __name__ == '__main__':
    for i in range(0, 10):
        j = "%06d" % (i + 1)
        lines = str + j + ".jpg --conf 0.3 --nms 0.65 --tsize 640 --save_result --device gpu"  # python命令 + B.py + 参数：IC.txt'
        print(lines)
        os.system(lines)
#str="python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_s.py -c YOLOX_outputs/yolox_voc_s/latest_ckpt.pth --path ./datasets/VOCdevkit/VOC2007/JPEGImages/000001.jpg --conf 0.3 --nms 0.65 --tsize 640 --save_result --device gpu"
#os.system(str)
#str1="python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_s.py -c YOLOX_outputs/yolox_voc_s/latest_ckpt.pth --path ./datasets/VOCdevkit/VOC2007/JPEGImages/000002.jpg --conf 0.3 --nms 0.65 --tsize 640 --save_result --device gpu"
#os.system(str1)
