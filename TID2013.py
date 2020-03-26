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

from .Dataset import IQADataset
from .utils import listt

def get_index(x): return re.findall(r'^([iI]\d{2})_(\d{2})_(\d).\w{3}', x)


class TID2013(IQADataset):
    INDEX_TYPE = 'MOS'
    INDEX_RANGE = np.array([0.0, 9.0])
    def generate_metafile(self, metafile_path):
        dataset_dir = self.dataset_dir
        original_imgs = os.listdir(join(dataset_dir, 'reference_images'))
        meta_file = list(csv.reader(open(join(dataset_dir, 'mos_with_names.txt')), delimiter=' '))
        meta_file = listt(meta_file)
        std_file = list(csv.reader(open(join(dataset_dir, 'mos_std.txt'))))

        std = [item[0] for item in std_file]
        mos = meta_file[0]
        file_name = meta_file[1]
        original_img = []
        distoration_type = []
        distoration_level = []

        for item in file_name:
            extracted = get_index(item)[0]
            distoration_type.append(extracted[1])
            distoration_level.append(extracted[2])
            for element in original_imgs:
                if extracted[0][1:3] == element[1:3]:
                    original_img.append(element)
                    break

        meta_base = DataFrame()
        meta_base['REF'] = original_img
        meta_base['REF_PATH'] = [join('reference_images', item) for item in original_img]
        meta_base['DIS'] = file_name
        meta_base['DIS_PATH'] = [join('distorted_images', item) for item in file_name]
        meta_base['INDEX'] = [float(item) for item in mos]
        meta_base['STD'] = [float(item) for item in std]
        meta_base['TYPE'] = distoration_type
        meta_base['LEVEL'] = distoration_level

        meta_base.to_pickle(metafile_path)
