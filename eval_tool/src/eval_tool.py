import cv2 as cv
import numpy as np
from typing import Tuple
import json
import os
import itertools


DATA_SET_FILEPATH_PREFIX: str = 'dataset/'
MAIS_FOLDERNAME: str = 'mais'
DATA_FOLDERNAMES = (MAIS_FOLDERNAME, )
JSON_FILEENDING: str = '.json'


#Supports ORB, AKAZE

Point = Tuple[int, int]
ROI = Tuple[Point, Point]

def get_features_ROI(ft_des: cv.Feature2D, img: np.ndarray, roi: ROI) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(img.shape[:2], np.uint8)
    mask = cv.rectangle(mask, roi[0], roi[1], (255), cv.FILLED)
    kp, des = ft_des.detectAndCompute(img, mask)
    return kp, des


def load_dataset(dataset_filename: str) -> dict:
    with open(dataset_filename) as data_fp:
        return json.load(data_fp)


def load_annotated_data() -> list:
    l = []
    for foldername in DATA_FOLDERNAMES:
        files = os.listdir(DATA_SET_FILEPATH_PREFIX + foldername)
        itertools.filterfalse(lambda x: x.endswith(JSON_FILEENDING), files)
        print(files)
        for x in files:
            with open(DATA_SET_FILEPATH_PREFIX + foldername + '/' + x) as json_file:
                json_data = json.load(json_file)
                l.append(json_data)

    return l

load_annotated_data()