from IQADataset.Dataset import IQADataset
import pandas as pd
from os.path import join, sep
import pickle
import numpy as np

SRC_DIR = 'src_imgs'
DST_DIR = 'dst_imgs'
META = 'csiq.DMOS.xlsx'


def get_dst_path(dis_type):
    if dis_type == 'noise':
        return 'awgn'
    elif dis_type == 'jpeg 2000':
        return 'jpeg2000'
    else: return dis_type



def get_dst_label(dis_type):
    upper = ['awgn', 'blur', 'jpeg']
    if dis_type in upper:
        return dis_type.upper()
    else:
        return dis_type


class CSIQ(IQADataset):
    INDEX_TYPE = 'DMOS'
    INDEX_RANGE = np.array([0.0, 1.0], dtype=np.float32)

    def generate_metafile(self, metafile_path):
        metadata = pd.DataFrame(columns='REF REF_PATH DIS_PATH TYPE INDEX STD'.split())
        with open(join(self.dataset_dir, META), 'rb') as f:
            data = pd.read_excel(f, sheet_name=6, header=3, usecols=range(3, 12))
        metadata.REF = data.image.apply(str) + '.png'
        metadata.REF_PATH = SRC_DIR + sep + metadata.REF
        metadata.DIS_PATH = DST_DIR \
                            + sep + data.dst_type.apply(get_dst_path) \
                            + sep + \
                            data.image.apply(str) + '.' + \
                            data.dst_type.apply(str).apply(get_dst_path).apply(get_dst_label) + '.' + \
                            data.dst_lev.apply(str) + '.png'
        metadata.INDEX = data.dmos
        metadata.TYPE = data.dst_type.apply(get_dst_path)
        metadata.STD = data.dmos_std
        with open(metafile_path, 'wb') as f:
            pickle.dump(metadata, f)


if __name__ == '__main__':
    dataset = CSIQ('/Users/akira/Dataset/CSIQ_ORIGINAL', using_data_pack=True)
    print(dataset[0])

