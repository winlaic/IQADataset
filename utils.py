from random import randint
from collections import namedtuple
from scipy.stats import spearmanr, pearsonr, kendalltau
import os
from os.path import join
import numpy as np
from PIL import Image


vector = namedtuple('vector', ['x', 'y'])

def listt(l): return list(map(list, zip(*l)))

def ensuredir(*args, file_name=None):
    path = join(*args)
    if not os.path.exists(path): 
        os.makedirs(path)
    elif not os.path.isdir(path):
        raise FileExistsError
    if file_name is not None:
        path = join(path, file_name)
    return path


IQAPerformance = namedtuple('IQAPerformance', 'SROCC PLCC MSE KROCC'.split())

def calculate_iqa_performace(input, target):
    return IQAPerformance(
        SROCC = float(spearmanr(input, target)[0]),
        PLCC  = float(pearsonr(input, target)[0]),
        MSE = float(np.mean((input - target)**2)),
        KROCC = float(kendalltau(input, target)[0])
    )



