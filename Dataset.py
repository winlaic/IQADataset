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
from torchvision.transforms import Compose, RandomHorizontalFlip
from .utils import LazyRandomCrop
from os.path import join
from .utils import ensuredir
from itertools import chain



class IQADataset(Dataset):
    """IQA Database base class. 
    For general purpose, only implement method 'generate_metafile' is sufficient.
    Besides, 'INDEX_TYPE' and 'INDEX_RANGE' must be specified.
    'INDEX_TYPE' must be one of 'MOS' or 'DMOS'.
    'INDEX_RANGE' must be numpy.ndarray with length of 2.
    """
    INDEX_TYPE = None
    INDEX_RANGE = None
    def __init__(self, dataset_dir, n_fold=5, crop_shape=None, random_flip=False, require_ref=False, using_data_pack=False, datapack_path='datapacks', metadata_path='metadata'):
        super().__init__()
        self.METAFILE = self.__class__.__name__ + '_metadata.pth'
        self.PARTITION_FILE = self.__class__.__name__ + '_partition.pth'
        self.dataset_dir = dataset_dir
        assert isinstance(n_fold, int) and n_fold >= 2
        self.n_fold = n_fold
        self.i_fold = -1
        self.use_augment = True # When true, argument transforms will be applied before output.
        self.require_ref = require_ref
        self.random_cropper = None
        self.metadata_path = metadata_path
        self.using_datapack = using_data_pack
        self.datapack_path = datapack_path
        self.DATAPACK = self.__class__.__name__ + '.pkl'
        self.datapack = None

        self.index_remapped = False
        self.__remap_k = 1
        self.__remap_c = 0
        
        assert self.INDEX_TYPE == 'MOS' or self.INDEX_TYPE == 'DMOS', 'Index type error.'
        assert isinstance(self.INDEX_RANGE, np.ndarray) and \
            len(self.INDEX_RANGE) == 2 and \
            self.INDEX_RANGE[0] < self.INDEX_RANGE[1], 'Index range error.'

        augment_transforms = []
        if crop_shape is not None:
            self.random_cropper = LazyRandomCrop(crop_shape)
            augment_transforms.append(self.random_cropper)
        if random_flip:
            augment_transforms.append(RandomHorizontalFlip(p=0.5))
        self.augment_transforms = Compose(augment_transforms)
        
        ensuredir(self.metadata_path)

        if not os.path.exists(join(self.metadata_path, self.METAFILE)):
            self.generate_metafile(join(self.metadata_path, self.METAFILE))
        self.metadata = pd.read_pickle(join(self.metadata_path, self.METAFILE))
        self.check_file_existance()

        if not os.path.exists(join(self.metadata_path, self.PARTITION_FILE)):
            self.divide_dataset(self.n_fold, join(self.metadata_path, self.PARTITION_FILE))
        self.partition_info = pickle.load(open(join(self.metadata_path, self.PARTITION_FILE), 'rb'))
        assert isinstance(self.partition_info, list), 'Partition info must be list.'

        if self.using_datapack:
            ensuredir(self.datapack_path)
            if not os.path.exists(join(self.datapack_path, self.DATAPACK)):
                self.generate_datapack()
            else:
                print('Loading datapack...', end = '', flush=True)
                with open(join(self.datapack_path, self.DATAPACK), 'rb') as f:
                    self.datapack = pickle.load(f)
                print('Done.')


        if len(self.partition_info) != self.n_fold:
            print('Warning: Partition file regenerated.')
            self.divide_dataset(self.n_fold, join(self.metadata_path, self.PARTITION_FILE))
            self.partition_info = pickle.load(open(join(self.metadata_path, self.PARTITION_FILE), 'rb'))

        self.train()

    def __remap_function(self, x):
        return self.__remap_k * x + self.__remap_c

    def remap_index(self, low, high):
        self.index_remapped = True
        self.__remap_k = (high - low) / (self.INDEX_RANGE[1] - self.INDEX_RANGE[0])
        self.__remap_c = low - self.INDEX_RANGE[0] * self.__remap_k

    def set_i_fold(self, i):
        """Use fold i as evaluating part.
        """
        assert i >= 0 and i < self.n_fold
        self.i_fold = i

    def train(self, not_on=None, use_augment=True, **kwargs):
        self._phase = 'train'
        self.use_augment = use_augment
        if not_on is None:
            not_on = self.i_fold

        deprecated_images = self.partition_info[not_on]
        self.generate_data_frame(deprecated_images = deprecated_images, **kwargs)
    
    def eval(self, use_augment=False, on=None, **kwargs):
        self._phase = 'eval'
        self.use_augment = use_augment
        deprecated_images = []

        if on is None:
            on = self.i_fold

        if on == -1:
            for item in self.partition_info[:-1]:
                deprecated_images += item
        else:
            for item in self.partition_info[:on]:
                deprecated_images += item
            for item in self.partition_info[on+1:]:
                deprecated_images += item
        self.generate_data_frame(deprecated_images = deprecated_images, **kwargs)

    def all(self, use_augment=False, **kwargs):
        self._phase = 'all'
        self.use_augment = use_augment
        self.generate_data_frame(**kwargs)

    def generate_datapack(self):
        datapack = {}
        print('Generating datapack...')
        total = len(self.metadata)
        for i, row in self.metadata.iterrows():
            print('{}%'.format(int(i/total*100.0)), end='\r', flush=True)
            img = cv.imread(join(self.dataset_dir, row.DIS_PATH))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            datapack[row.DIS_PATH] = img
        ref_file_paths = self.metadata.REF_PATH.unique().tolist()
        for ref in ref_file_paths:
            img = cv.imread(join(self.dataset_dir, ref))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            datapack[ref] = img
        print('Dumping...', end='', flush=True)
        with open(join(self.datapack_path, self.DATAPACK), 'wb') as f:
            pickle.dump(datapack, f)
        print('Done')
        self.datapack = datapack
    

    def generate_metafile(self, metafile_path):
        """Generate metafile for the whole database.
        This method must be implement.
        You should complete the following steps in this method:
        1. Construct a pandas DataFrame with at least these properties for each image.
            REF REF_PATH DIS_PATH TYPE INDEX
        2. Extra information:
            LEVEL STD DIS
        3. Before return, save the dataframe to 'metafile_path'.
        """
        raise NotImplementedError

    def divide_dataset(self, n_fold, divide_metafile_path):
        ref_imgs = self.metadata['REF'].unique()
        n_all = len(ref_imgs)
        n_per_fold = n_all // n_fold

        indexes = np.random.choice(n_all, n_all, replace=False)
        partitioned_indexes = np.split(indexes, range(n_per_fold, n_all, n_per_fold))
        partition_list = []
        for i, incide in enumerate(partitioned_indexes):
            if i != self.n_fold:
                partition_list.append(ref_imgs[incide].tolist())
            else:
                last_part = ref_imgs[incide].tolist()
                for j, single in enumerate(last_part):
                    partition_list[j].append(single)
        
        with open(divide_metafile_path, 'wb') as f:
            pickle.dump(partition_list, f)

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

    @staticmethod
    def preprocess(img):
        img = img.astype(np.float32)/255.0
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img)
        return img

    def check_file_existance(self):
        ref_imgs = self.metadata.REF_PATH.unique()
        dis_imgs = self.metadata.DIS_PATH.to_list()
        for item in chain(ref_imgs, dis_imgs):
            assert os.path.exists(join(self.dataset_dir, item)), 'Image "{}" does not exist!'.format(item)

    
    def __getitem__(self, index):
        if index >= len(self): raise IndexError
        meta = self.__data_frame[index:index+1]

        def load_img(dir, path, datapack=None):
            if datapack is not None:
                img = self.datapack[path]
            else:
                img = cv.imread(join(dir, path))
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = Image.fromarray(img)
            if self.use_augment:
                img = self.augment_transforms(img)
            img = np.array(img)
            return img
        
        img = load_img(self.dataset_dir, meta.DIS_PATH.to_list()[0], self.datapack)
        img = self.preprocess(img)
        if self.require_ref:
            img_ref = load_img(self.dataset_dir, meta.REF_PATH.to_list()[0], self.datapack)
            img_ref = self.preprocess(img_ref)


        label = torch.tensor(meta.INDEX.to_list()[0])
        if self.index_remapped:
            label = self.__remap_function(label)
        dis_type = meta.TYPE.to_list()[0]

        if self.random_cropper: self.random_cropper.reset()

        if self.require_ref:
            return img, img_ref, label, dis_type
        else:
            return img, label, dis_type
            

        
    def __len__(self):
        return len(self.__data_frame)

