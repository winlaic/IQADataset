from .Dataset import IQADataset
from scipy.io import loadmat
import numpy as np
from pandas import DataFrame
from os.path import join


class LIVEChallengeDB(IQADataset):
    INDEX_TYPE = 'MOS'
    INDEX_RANGE = np.array([0, 100])
    def generate_metafile(self, metafile_path):
        DATABASE_DIR = self.dataset_dir
        IMAGE_META = 'Data/AllImages_release.mat'
        MOS_META = 'Data/AllMOS_release.mat'
        STD_META = 'Data/AllStdDev_release.mat'
        img_names = loadmat(join(DATABASE_DIR, IMAGE_META))['AllImages_release']
        img_names = list(map(lambda x: str(x[0][0]), img_names))
        img_types = list(map(lambda x: 'for_train' if x[0] == 't' else 'real_world', img_names))
        img_pathes = [join('Images', item) if item[0] != 't' else join('Images', 'trainingImages', item) for item in img_names]
        img_dummy_refs = ['dummy_' + ''.join(item.split('.')[:-1]) for item in img_names]
                
        mos = loadmat(join(DATABASE_DIR, MOS_META))['AllMOS_release'].squeeze().tolist()
        std = loadmat(join(DATABASE_DIR, STD_META))['AllStdDev_release'].squeeze().tolist()
        dataframe = DataFrame()
        dataframe['DIS_PATH'] = img_pathes
        dataframe['REF_PATH'] = img_dummy_refs
        dataframe['REF'] = img_dummy_refs
        dataframe['INDEX'] = mos
        dataframe['TYPE'] = img_types
        dataframe['STD'] = std
        dataframe.to_pickle(metafile_path)

    def generate_data_frame(self, *args, **kwargs):
        if 'include_ref' in kwargs:
            assert not kwargs['include_ref'], 'For LIVE Challenge Database, there are no refer images.'
        super().generate_data_frame(*args, **kwargs)