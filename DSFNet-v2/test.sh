# python test.py --model_name DSFNet --gpus 0 \
python testTrackingSort.py --model_name DSFNet --gpus 0 \
--load_model ./weights/rsdata/DSFNet/viso-car/model_best.pth \
--test_large_size True --datasetname rsdata --data_dir  ./data/viso-test/ \
--save_track_results True