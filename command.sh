# ignore error
set TF_ENABLE_ONEDNN_OPTS=0

# tensorboard
tensorboard --logdir results/logs

# train: python train.py --video_path data/{video}.mp4 --is_masking {T/F} --pipe_num {#} --omnet_mode {T/F}

python train.py --video_path data/RoadVideo_train.mp4
python train.py --video_path data/RoadVideo_train.mp4 --is_masking False --pipe_num 2


# test: python run.py --video_path data/{video}.mp4 --model_path models/ndarray/{model}.npy  --is_masking {T/F} --output_path results/{output}.mp4  
python run.py --video_path data/RoadVideo_test.mp4 --model_path models/ndarray/240130-151415_videopath_RoadVideotrain_ismasking_False.npy  --is_masking False --output_path results/SLN_False_False.mp4 --V 50
python run.py --video_path data/RoadVideo_test.mp4 --model_path models/ndarray/240130-151415_videopath_RoadVideotrain_ismasking_False.npy  --is_masking True --output_path results/SLN_False_True.mp4 --V 50 --pipe_num 2
python run.py --video_path data/RoadVideo_test.mp4 --model_path models/ndarray/240130-151415_videopath_RoadVideotrain_ismasking_False.npy  --is_masking False --output_path results/YOLO_False_False.mp4  --V 10000
python run.py --video_path data/RoadVideo_test.mp4 --model_path models/ndarray/240130-151415_videopath_RoadVideotrain_ismasking_False.npy  --is_masking True --output_path results/YOLO_False_True.mp4  --V 10000 --pipe_num 2