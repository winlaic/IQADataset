# Developed by winlaic in Wuhan University.
# 2019.11.21 03:38

from torch.utils.data import Dataset
import torch, numpy as np
import os
from os.path import join
import pandas as pd
import cv2 as cv
from PIL import Image
import pickle
from pandas import DataFrame
from torchvision.transforms import RandomCrop, Compose, RandomHorizontalFlip

'''
Metadata should have these information.
0. Original image path.
1. Target image path.
2. Distoration type.
3. MOS/DMOS
4. Std.
'''


class IQADataset(Dataset):
    """IQA Database base class. 
    For general purpose, only implement method 'generate_metafile' is sufficient.
    Besides, 'INDEX_TYPE' and 'INDEX_RANGE' must be specified.
    'INDEX_TYPE' must be one of 'MOS' or 'DMOS'.
    'INDEX_RANGE' must be numpy.ndarray with length of 2.
    """
    INDEX_TYPE = None
    INDEX_RANGE = None
    def __init__(self, dataset_dir, train_ratio=0.8, crop_shape=None, random_flip=False):
        super().__init__()
        self.METAFILE = self.__class__.__name__ + '_metadata.pth'
        self.PARTITION_FILE = self.__class__.__name__ + '_partition.pth'
        self.dataset_dir = dataset_dir
        self.train_ratio = train_ratio
        self.augment = True

        self.index_remapped = False
        self.__remap_function = lambda x: x
        
        assert self.INDEX_TYPE == 'MOS' or self.INDEX_TYPE == 'DMOS', 'Index type error.'
        assert isinstance(self.INDEX_RANGE, np.ndarray) and \
            len(self.INDEX_RANGE) == 2 and \
            self.INDEX_RANGE[0] < self.INDEX_RANGE[1], 'Index range error.'

        augment_transforms = []
        if crop_shape is not None:
            augment_transforms.append(RandomCrop(crop_shape))
        if random_flip:
            augment_transforms.append(RandomHorizontalFlip(p=0.5))
        self.augment_transforms = Compose(augment_transforms)

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

    def remap_index(self, low, high):
        self.index_remapped = True
        a = (high - low) / (self.INDEX_RANGE[1] - self.INDEX_RANGE[0])
        c = low - self.INDEX_RANGE[0] * a
        self.__remap_function = lambda x: a*x + c

    def train(self, augment=True, **kwargs):
        self._phase = 'train'
        self.augment = augment
        self.generate_data_frame(deprecated_images = self.partition_info['val'], **kwargs)
    
    def eval(self, **kwargs):
        self._phase = 'eval'
        self.generate_data_frame(deprecated_images = self.partition_info['train'], **kwargs)

    def all(self, **kwargs):
        self._phase = 'all'
        self.generate_data_frame(**kwargs)
    

    def generate_metafile(self, metafile_path):
        """Generate metafile for the whole database.
        This method must be implement.
        You should complete the following steps in this method:
        1. Construct a pandas DataFrame with at least these properties for each image.
            REF REF_PATH DIS_PATH TYPE INDEX
        2. Before return, save the dataframe to 'metafile_path'.
        """
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
            if self.INDEX_TYPE == 'DMOS':
                ref_imgs_data_frame['INDEX'] = np.min(self.INDEX_RANGE)
            elif self.INDEX_TYPE == 'MOS':
                ref_imgs_data_frame['INDEX'] = np.max(self.INDEX_RANGE)
            ref_imgs_data_frame['TYPE'] = 'pristine'

            self.__data_frame = self.__data_frame.append(ref_imgs_data_frame, ignore_index=True, sort=False)

    def preprocess(self, img):
        img = img.astype(np.float32)/255.0
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img)
        return img
    
    def __getitem__(self, index):
        if index >= len(self): raise IndexError
        meta = self.__data_frame[index:index+1]
        img = cv.imread(join(self.dataset_dir, meta.DIS_PATH.to_list()[0]))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if (self._phase == 'train') and self.augment:
            img = self.augment_transforms(img)
        img = np.array(img)
        img = self.preprocess(img)
        label = torch.tensor(meta.INDEX.to_list()[0])
        if self.index_remapped:
            label = self.__remap_function(label)
        dis_type = meta.TYPE.to_list()[0]
        return img, label, dis_type

        
    def __len__(self):
        return len(self.__data_frame)