# Developed by winlaic in Wuhan University.
# 2019.11.21 03:38

from torch.utils.data import Dataset
import torch, numpy as np
import os
from os.path import join
import pandas as pd
import cv2 as cv
import pickle
from pandas import DataFrame

'''
Metadata should have these information.
0. Original image path.
1. Target image path.
2. Distoration type.
3. MOS/DMOS
4. Std.
'''


class IQADataset(Dataset):
    
    def __init__(self, dataset_dir, train_ratio=0.8):
        super().__init__()
        self.METAFILE = self.__class__.__name__ + '_metadata.pth'
        self.PARTITION_FILE = self.__class__.__name__ + '_partition.pth'
        self.dataset_dir = dataset_dir
        self.train_ratio = train_ratio

        if not os.path.exists(self.METAFILE):
            self.generate_metafile(self.METAFILE)
        self.metadata = pd.read_pickle(self.METAFILE)

        if not os.path.exists(self.PARTITION_FILE):
            self.divide_dataset(self.train_ratio, self.PARTITION_FILE)
        self.partition_info = pickle.load(open(self.PARTITION_FILE, 'rb'))

        if self.partition_info['ratio'] != train_ratio:
            print('Warning: Partition file regenerated.')
            self.divide_dataset(self.train_ratio, self.PARTITION_FILE)
            self.partition_info = pickle.load(open(self.PARTITION_FILE, 'rb'))

        self.train()


    def train(self, **kwargs):
        self.__phase = 'train'
        self.generate_data_frame(deprecated_images = self.partition_info['val'], **kwargs)
    
    def eval(self, **kwargs):
        self.__phase = 'eval'
        self.generate_data_frame(deprecated_images = self.partition_info['train'], **kwargs)

    def all(self, **kwargs):
        self.__phase = 'all'
        self.generate_data_frame(**kwargs)
    

    def generate_metafile(self, metafile_path):
        raise NotImplementedError

    def divide_dataset(self, divide_ratio, divide_metafile_path):
        ref_imgs = self.metadata['REF'].unique()
        n_all = len(ref_imgs)
        partition_dict = {}
        partition_dict['ratio'] = divide_ratio
        n_train = int(n_all * divide_ratio)
        # n_val = n_all - n_train
        indexes = np.random.choice(n_all, n_all, replace=False)
        partition_dict['train'] = ref_imgs[indexes[:n_train]]
        partition_dict['val'] = ref_imgs[indexes[n_train:]]
        with open(divide_metafile_path, 'wb') as f:
            pickle.dump(partition_dict, f)

    def generate_data_frame(self, deprecated_types = [], deprecated_images = [], include_ref=False):
        self.__deprecated_types = deprecated_types
        self.__deprecated_images = deprecated_images
        self.__data_frame = self.metadata[
            -(self.metadata.REF.isin(deprecated_images) | self.metadata.TYPE.isin(deprecated_types))]
        if include_ref:
            ref_imgs_data_frame = DataFrame()
            ref_imgs_data_frame['DIS_PATH'] = self.__data_frame['REF_PATH'].unique()
            ref_imgs_data_frame['INDEX'] = 0.0 if 'LIVE' in self.__class__.__name__ else 9.0
            ref_imgs_data_frame['TYPE'] = 'pristine'

            self.__data_frame = self.__data_frame.append(ref_imgs_data_frame, ignore_index=True, sort=False)



    def __getitem__(self, index):
        if index >= len(self): raise IndexError
        meta = self.__data_frame[index:index+1]
        img = cv.imread(join(self.dataset_dir, meta.DIS_PATH.to_list()[0]))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.0
        img = img.transpose(2, 0, 1)
        return torch.tensor(img), torch.tensor(meta.INDEX.to_list()[0]), meta.TYPE.to_list()[0]

        
    def __len__(self):
        return len(self.__data_frame)