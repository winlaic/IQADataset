# Developed by winlaic in Wuhan University.
# 2019.11.21 03:38

import csv
import os
import pickle
import re
import types
from os.path import join

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.io import loadmat

from .Dataset import IQADataset

listt = lambda l: list(map(list, zip(*l)))

class LIVE2016(IQADataset):

    def generate_metafile(self, metafile_path):
        dataset_dir = self.dataset_dir
        distoration_classes = ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading']
        distoration_image_number = [227, 233, 174, 174, 174]
        # reference_imgs = [item for item in os.listdir(join(dataset_dir, 'refimgs')) if item.split(sep='.')[-1] == 'bmp']

        listt = lambda x: list(map(list, zip(*x)))
        dmos_of_distoration_types = []
        is_original_image = []
        labels = loadmat(join(dataset_dir, 'dmos.mat'))

        for i in range(len(distoration_image_number)):
            previous_index = sum(distoration_image_number[0:i])
            next_index = previous_index + distoration_image_number[i]
            dmos_of_distoration_types.append(labels['dmos'].squeeze()[previous_index:next_index].tolist())
            is_original_image.append(labels['orgs'].squeeze()[previous_index:next_index].tolist())
        
        infos = []
        for i, item in enumerate(distoration_classes):
            with open(join(dataset_dir, item, 'info.txt')) as f:
                info = list(csv.reader(f, delimiter=' '))
                info = [iitem for iitem in info if len(iitem) != 0]
                info = listt(listt(info)[0:2])
                info.sort(key=lambda x: int(x[1].split(sep='.')[0][3:]))
                info = listt(info)
                info.append(dmos_of_distoration_types[i])
                info.append(is_original_image[i])
                info.insert(1, list(map(lambda x: join('refimgs', x), info[0])))
                info[2] = list(map(lambda x: join(distoration_classes[i], x), info[2]))
                info.insert(1, [item for _ in range(len(info[0]))])
                info = listt(info)
                infos += info


        infos_distored_only = [item for item in infos if item[-1]==0]
        infos_distored_only.sort(key=lambda x: x[0])
        infos_distored_only = listt(infos_distored_only)

        meta_base = DataFrame()
        meta_base['REF'] = infos_distored_only[0]
        meta_base['REF_PATH'] = infos_distored_only[2]
        meta_base['DIS_PATH'] = infos_distored_only[3]
        meta_base['INDEX'] = [float(item) for item in infos_distored_only[4]]
        meta_base['TYPE'] = infos_distored_only[1]

        meta_base.to_pickle(metafile_path)
