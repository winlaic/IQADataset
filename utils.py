from random import randint
from collections import namedtuple
from scipy.stats import spearmanr, pearsonr, kendalltau
import os
from os.path import join

vector = namedtuple('vector', ['x', 'y'])


def ensuredir(*args, file_name=None):
    path = join(*args)
    if not os.path.exists(path): 
        os.makedirs(path)
    elif not os.path.isdir(path):
        raise FileExistsError
    if file_name is not None:
        path = join(path, file_name)
    return path

class LazyRandomCrop():
    """Crop position does not change untile "reset()" called or shape of input image changed.
    """
    def __init__(self, size):
        self.size = vector._make((int(size),) * 2)
        self.prevous_shape = None
        self.position = None

    def reset(self):
        self.prevous_shape = None
        self.position = None


    def __call__(self, img):
        if self.prevous_shape is None:
            self.prevous_shape = vector._make(img.shape)
            self.position = vector(randint(0, self.prevous_shape.x - self.size.x), randint(0, self.prevous_shape.y - self.size.y))
        elif self.prevous_shape != img.shape:
            print('LazyRandomCropper: Input image shape changed without calling "reset()".')
            self.prevous_shape = vector(*img.shape)
            self.position = vector(randint(0, self.prevous_shape.x - self.size.x), randint(0, self.prevous_shape.y - self.size.y))
        position_lu = self.position
        position_rd = vector(position_lu.x + self.size.x, position_lu.y + self.size.y)
        return img[position_lu.x: position_rd.x, position_lu.y: position_rd.y]

    def __repr__(self):
        return '{}: Crop Size={}, Previous shape={}, Crop position={}'.format(self.__class__.__name__, self.size, self.prevous_shape, self.position)

IQAPerformance = namedtuple('IQAPerformance', ['SROCC', 'PLCC', 'KROCC'])

def calculate_iqa_performace(input, target):
    return IQAPerformance(
        SROCC = float(spearmanr(input, target)[0]),
        PLCC  = float(pearsonr(input, target)[0]),
        KROCC = float(kendalltau(input, target)[0])
    )

        


