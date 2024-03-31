import os




searchdir = "D:/VSC/INFOCOM/FrameDrop-RL/models/ndarray/new"

for entry in os.listdir(searchdir):
    if os.path.isfile(os.path.join(searchdir, entry)):
        filename = os.path.basename(entry)
        filename_without_type = filename[:-4]
        conf_list = filename.split('_')
        date = conf_list[0]
        dataset = conf_list[2]
        
        if date > "240328-140300":
            if dataset == "JK":
                test_dataset = ["JK-1"]
            elif dataset == "SD":
                test_dataset = ["SD-1"]
            elif dataset == "JN":
                test_dataset = ["JN"]
            
            for data in test_dataset:
                command = f"python run.py -video ../DATASET/test/{data}.mp4 -model models/ndarray/new/{filename} -mask f -f1 t"
                os.system(command)