# tensorboard
tensorboard --logdir results\logs

# train
python train.py --video_path data\RoadVideo_train.mp4
python train.py --video_path data\RoadVideo_train.mp4 --is_masking False --pipe_num 2