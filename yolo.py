from utils.yolov5.detect import inference

command = ["--weights", "models/yolov5s6.pt", "--source", 'C:/Users/s_bono0208/Downloads/test.mp4', "--project", 'yolo/results/', "--name", 'test', "--save-txt", "--save-conf"]
inference(command)