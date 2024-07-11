python train.py --model_name DSFNet --gpus 0 --lr 1.25e-4 \
--lr_step 30,45 --num_epochs 55 --batch_size 1 --val_intervals 5  \
--test_large_size True --datasetname rsdata --data_dir  ./data/viso-ship/

python testTrackingSort.py --model_name DSFNet --gpus 0 \
--load_model ./weights/rsdata/DSFNet/viso-car-1/model_best.pth \
--test_large_size True --datasetname rsdata --data_dir  ./data/viso-ship/ \
--save_track_results True