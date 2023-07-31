
import os

def _detect(resultPath, numFrame):
        fileList =  os.listdir(resultPath)
        idx = 1
        for fileName in fileList :
            head = fileName.split("_")[0]
            temp = fileName.split("_")[-1]
            fileIdx = int(temp.split(".")[0])
            while fileIdx > idx :
                filePath = resultPath+"/"+head+"_"+str(idx)+".txt"
                with open(filePath, 'w') as file :
                    pass
                idx+=1
            idx+=1

_detect("utils/yolov5/runs/detect/exp4/labels", 8546)