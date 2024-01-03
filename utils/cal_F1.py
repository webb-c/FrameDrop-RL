import os
"""
for calculate F1 score;
using ref_frame comparison with skip_frame
"""

def _parse_results(filename):
    with open(filename, 'r') as file :
        labels = []
        for line in file :
            line = line.strip().split()
            x, y, w, h = map(float, line[1:5])
            labels.append((x, y, w, h))
    return labels

def _cal_F1(filePred, skipFilePred, threshold=0.5) :  # pred, labels
    TP = 0
    FP = 0
    FN = 0
    
    for fPred in filePred :
        fx, fy, fw, fh = fPred
        fArea = fw*fh
        
        maxIOU = 0.0
        for sPred in skipFilePred :
            sx, sy, sw, sh = sPred
            sArea = sw*sh
            
            interArea = max(0, min(fx+fw, sx+sw) - max(fx, sx)) * max(0, min(fy+fh, sy+sh) - max(fy, sy))
            unionArea = fArea + sArea - interArea
            IOU = interArea / unionArea
            maxIOU = max(maxIOU, IOU)
        
        if maxIOU >= threshold :
            TP += 1
        else :
            FP += 1
    
    FN = len(skipFilePred) - TP
    if TP + FP == 0: precision = 0
    else: precision = TP / (TP+FP)
    if TP + FN == 0: recall = 0
    else: recall = TP /  (TP+FN)
    F1 = 2*(precision*recall) / (precision+recall) if (precision+recall) != 0 else 0
    return F1

def get_F1(fileName, skipFileName) :
    filePred = _parse_results(fileName)
    skipFilePred = _parse_results(skipFileName)
    return _cal_F1(filePred, skipFilePred)