# python3 ./tools/train.py -f ./exps/example/yolox_voc/yolox_voc_s.py -d 4 -b 12 -c yolox_s.pth 2>&1 | tee log.txt
# python3 ./tools/train.py -f ./exps/example/yolox_voc/yolox_voc_l.py -d 4 -b 8 -c yolox_l.pth 2>&1 | tee log.txt
python3 ./tools/train.py -f ./exps/example/yolox_voc/yolox_voc_x.py -d 4 -b 12 -c yolox_x.pth 2>&1 | tee log.txt
# python3 ./tools/train.py -f ./exps/example/yolox_voc/yolox_voc_x.py -d 4 -b 12 -c ./YOLOX_outputs/yolox_voc_x-other/best_ckpt.pth
# python tools/train.py -n yolox-x -d 1 -b 32 --fp16 -o --cache