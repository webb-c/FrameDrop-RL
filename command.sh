# ignore error
set TF_ENABLE_ONEDNN_OPTS=0

# tensorboard
tensorboard --logdir results/logs

# train: python train.py --video_path data/{video}.mp4 --is_masking {T/F} --pipe_num {#} --omnet_mode {T/F}
python train.py --video_path data/RoadVideo_train.mp4
python train.py --video_path data/RoadVideo_train.mp4 --is_masking False --pipe_num 2


# test: python run.py --video_path data/{video}.mp4 --model_path models/ndarray/{model}.npy  --is_masking {T/F} --output_path results/{output}.mp4  


# V value
python run.py --video_path data/RoadVideo_test.mp4 --model_path models/ndarray/240130-151415_videopath_RoadVideotrain_ismasking_False.npy --is_masking False --f1_score False --output_path results/ABILENE_False_False_Agent1.mp4 --V 1000000
python run.py --video_path data/RoadVideo_test.mp4 --model_path models/ndarray/240130-151415_videopath_RoadVideotrain_ismasking_False.npy --is_masking False --f1_score False --output_path results/ABILENE_False_False_Agent2.mp4 --V 1000000 --pipe_num 2
python run.py --video_path data/RoadVideo_test.mp4 --model_path models/ndarray/240130-151415_videopath_RoadVideotrain_ismasking_False.npy --is_masking False --f1_score False --output_path results/ABILENE_False_False_Agent2.mp4 --V 1000000 --pipe_num 3


# Dataset
$ ../DATASET/train/JK-1.mp4 
$ ../DATASET/train/JK-2.mp4 
$ ../DATASET/train/JN.mp4 

$ ../DATASET/test/JK-1.mp4 
$ ../DATASET/test/JK-2.mp4 
$ ../DATASET/test/JN.mp4 


# Temporary PERFORMANCE INCREASING TEST
$ python train.py -video ../DATASET/train/{}.mp4 -pi {} -mask False -omnet False
$ python run.py -videp ../DATASET/test/{}.mp4 -model models/ndarray/{}.npy -out results/PI/{}.mp4 -mask False -omnet False -f1 True 

