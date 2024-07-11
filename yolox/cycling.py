import os
import multiprocessing
import time

def deal(start,end):
    for i in range(start,end+1):
        j="%06d"%i
        lines=str+j+".jpg --conf 0.3 --nms 0.65 --tsize 640 --save_result --device gpu"   #python命令 + B.py + 参数：IC.txt'
        #print(lines)
        os.system(lines)



if __name__ == '__main__':
    str = "python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_s.py -c YOLOX_outputs/yolox_voc_s/latest_ckpt.pth --path ./datasets/VOCdevkit/VOC2007/JPEGImages/"
    p2_1 = multiprocessing.Process(target=deal,args=(1359,1500))
    p2_2 = multiprocessing.Process(target=deal, args=(1501, 1647))
    p3 = multiprocessing.Process(target=deal, args=(290, 468))
    p5 = multiprocessing.Process(target=deal, args=(1648,1837))
    p9_1 = multiprocessing.Process(target=deal, args=(2218,2400))
    p9_2 = multiprocessing.Process(target=deal, args=(2401, 2600))
    p9_3 = multiprocessing.Process(target=deal, args=(2601, 2800))
    p9_4 = multiprocessing.Process(target=deal, args=(2801, 2917))

    p2_1.start()
    p2_2.start()
    p3.start()
    p5.start()
    p9_1.start()
    p9_2.start()
    p9_3.start()
    p9_4.start()

#os._exit()