import os
import numpy as np
import matplotlib.image as mpimg
import itertools
from collections import deque
import re

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str):
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

def sort_humanly(v_list):
    return sorted(v_list, key=str2int)


sourceDic = os.getcwd()
# put images in /Data/rawdata
dicList = os.listdir(sourceDic + '/Data/rawData')
if not os.path.exists(sourceDic + '/Data/processedData'):
    os.mkdir(sourceDic + '/Data/processedData')
if not os.path.exists(sourceDic + '/Data/processedData/train'):
    os.mkdir(sourceDic + '/Data/processedData/train')
if not os.path.exists(sourceDic + '/Data/processedData/val'):
    os.mkdir(sourceDic + '/Data/processedData/val')
if not os.path.exists(sourceDic + '/Data/processedData/test'):
    os.mkdir(sourceDic + '/Data/processedData/test')
valBound = 12000
testBound = 16000
Bound = 20000

countL = 0
countS = 0
countD = 0
L_label = False
S_label = False
D_label = False
frameList = deque(maxlen=20)
background = []
vessel = []
ori = []
for dicName in dicList:
    frameList.clear()
    background.clear()
    vessel.clear()
    ori.clear()
    fileList = os.listdir(sourceDic + '/Data/rawData/' + dicName)
    print(dicName)
    for fileName in fileList:
        if 'background' in fileName:
            background.append(fileName)
        elif 'vessel' in fileName:
            vessel.append(fileName)
        elif 'ori' in fileName:
            ori.append(fileName)

    background = sort_humanly(background)
    vessel = sort_humanly(vessel)
    ori = sort_humanly(ori)
    for fileName in background:
        img = mpimg.imread(sourceDic + '/Data/rawData/' + dicName + '/' + fileName)
        frameList.append(img[:, :, None])
        if len(frameList) == 20:
            videoData = np.concatenate(tuple(frameList), axis=2)
            height, width = img.shape
            heightAxisStart = [x for x in range(0, (height - 65), 32)]
            widthAxisStart = [x for x in range(0, (width - 65), 32)]
            StartPointList = [y for y in itertools.product(heightAxisStart, widthAxisStart)]
            for h, w in StartPointList:
                if countL < valBound:
                    np.save(sourceDic + '/Data/processedData/train/L_xca%.5d.npy' % countL,
                            videoData[h:h + 64, w:w + 64, :])
                elif countL < testBound:
                    np.save(sourceDic + '/Data/processedData/val/L_xca%.5d.npy' % (countL-valBound),
                            videoData[h:h + 64, w:w + 64, :])
                elif countL < Bound:
                    np.save(sourceDic + '/Data/processedData/test/L_xca%.5d.npy' % (countL-testBound),
                            videoData[h:h + 64, w:w + 64, :])
                else:
                    L_label = True
                    break
                countL += 1
            frameList.clear()
        if L_label: break
    frameList.clear()
    for fileName in vessel:
        img = mpimg.imread(sourceDic + '/Data/rawData/' + dicName + '/' + fileName)
        frameList.append(img[:, :, None])
        if len(frameList) == 20:
            videoData = np.concatenate(tuple(frameList), axis=2)
            height, width = img.shape
            heightAxisStart = [x for x in range(0, (height - 65), 32)]
            widthAxisStart = [x for x in range(0, (width - 65), 32)]
            StartPointList = [y for y in itertools.product(heightAxisStart, widthAxisStart)]
            for h, w in StartPointList:
                if countS < valBound:
                    np.save(sourceDic + '/Data/processedData/train/S_xca%.5d.npy' % countS,
                            videoData[h:h + 64, w:w + 64, :])
                elif countS < testBound:
                    np.save(sourceDic + '/Data/processedData/val/S_xca%.5d.npy' % (countS-valBound),
                            videoData[h:h + 64, w:w + 64, :])
                elif countS < Bound:
                    np.save(sourceDic + '/Data/processedData/test/S_xca%.5d.npy' % (countS-testBound),
                            videoData[h:h + 64, w:w + 64, :])
                else:
                    S_label = True
                    break
                countS += 1
            frameList.clear()
        if S_label: break
    frameList.clear()
    for fileName in ori:
        img = mpimg.imread(sourceDic + '/Data/rawData/' + dicName + '/' + fileName)
        frameList.append(img[:, :, None])
        if len(frameList) == 20:
            videoData = np.concatenate(tuple(frameList), axis=2)
            height, width = img.shape
            heightAxisStart = [x for x in range(0, (height - 65), 32)]
            widthAxisStart = [x for x in range(0, (width - 65), 32)]
            StartPointList = [y for y in itertools.product(heightAxisStart, widthAxisStart)]
            for h, w in StartPointList:
                if countD < valBound:
                    np.save(sourceDic + '/Data/processedData/train/D_xca%.5d.npy' % countD,
                            videoData[h:h + 64, w:w + 64, :])
                elif countD < testBound:
                    np.save(sourceDic + '/Data/processedData/val/D_xca%.5d.npy' % (countD-valBound),
                            videoData[h:h + 64, w:w + 64, :])
                elif countD < Bound:
                    np.save(sourceDic + '/Data/processedData/test/D_xca%.5d.npy' % (countD-testBound),
                            videoData[h:h + 64, w:w + 64, :])
                else:
                    D_label = True
                    break
                countD += 1
            frameList.clear()
        if D_label: break
    if D_label and S_label and L_label: break