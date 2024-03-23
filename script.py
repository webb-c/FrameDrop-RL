import os

searchdir = "D:/VSC/INFOCOM/FrameDrop-RL/models/ndarray"

for entry in os.listdir(searchdir):
    if os.path.isfile(os.path.join(searchdir, entry)):
        filename = os.path.basename(entry)
        filename_without_type = filename[:-4]
        conf_list = filename.split('_')
        date = conf_list[0]
        dataset = conf_list[2]
        last = conf_list[-1][:-4]
        is_epi = False
        try:
            last = float(last)
            if last.is_integer():
                is_epi = True
            else:
                is_epi = False
        except ValueError:
            is_epi = False
        
        if date > "240321-220451":
            if not is_epi:
                if dataset == "JK":
                    test_dataset = ["JK-1", "JK-2"]
                elif dataset == "SD":
                    test_dataset = ["SD-1", "SD-2"]
                elif dataset == "JN":
                    test_dataset = ["JN"]
                
                for data in test_dataset:
                    command = f"python run.py -video ../DATASET/test/{data}.mp4 -model models/ndarray/{filename} -out results/PI/{date}_{data}.mp4 -log {filename_without_type}_{data} -mask f -f1 t"
                    os.system(command)