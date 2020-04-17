import numpy as np
import torch
from numpy.lib.stride_tricks import as_strided
from scipy.signal import convolve2d
from PIL import Image
# from torchvision.transforms import *


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


    def __call__(self, img: Image):
        img = np.array(img)
        if self.prevous_shape is None:
            self.prevous_shape = vector._make(img.shape[:-1])
            self.position = vector(randint(0, self.prevous_shape.x - self.size.x), randint(0, self.prevous_shape.y - self.size.y))
        elif self.prevous_shape != img.shape[:-1]:
            print('LazyRandomCropper: Input image shape changed without calling "reset()".')
            self.prevous_shape = vector(*img.shape[:-1])
            self.position = vector(randint(0, self.prevous_shape.x - self.size.x), randint(0, self.prevous_shape.y - self.size.y))
        position_lu = self.position
        position_rd = vector(position_lu.x + self.size.x, position_lu.y + self.size.y)
        return Image.fromarray(img[position_lu.x: position_rd.x, position_lu.y: position_rd.y])

    def __repr__(self):
        return '{}: Crop Size={}, Previous shape={}, Crop position={}'.format(self.__class__.__name__, self.size, self.prevous_shape, self.position)


class LocalNormalization:
    
    def __init__(self, P=3, Q=3, C=1):
        self.P = P
        self.Q = Q
        self.C = C
    
    def __call__(self, patch):
        patch = np.array(patch)
        if len(patch.shape) == 3:
            assert patch.shape[2] == 1
        else:
            assert len(patch.shape) == 2
        P = self.P; Q = self.Q; C = self.C
        kernel = np.ones((P, Q)) / (P * Q)
        patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
        patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
        patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
        patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float()
        return patch_ln

class PatchExtractor:
    def __init__(self, patch_shape, strides, n_random_choice=None):
        super().__init__()
        if isinstance(patch_shape, int):
            self.patch_shape = (patch_shape,) * 2
        elif isinstance(patch_shape, (list, tuple)):
            assert len(patch_shape) == 2
            self.patch_shape = tuple(patch_shape)
        else:
            raise TypeError()

        if isinstance(strides, int):
            self.strides = (strides,) * 2
        elif isinstance(strides, (list, tuple)):
            assert len(strides) == 2
            self.strides = tuple(strides)
        else:
            raise TypeError()

        self.n_random_choice = n_random_choice

    def __call__(self, img):
        img = np.array(img)
        if len(img.shape) == 3:
            assert img.shape[2] == 3 or img.shape[2] == 1
        else:
            assert len(img.shape) == 2
        patches = extract_patches(img, self.patch_shape, self.strides)
        patches = patches.reshape(-1, *patches.shape[-3:])
        if self.n_random_choice is not None:
            inc = np.random.choice(patches.shape[0], self.n_random_choice, replace=False)
            patches = patches[inc, :, :, :]
        return patches



        

def extract_patches(img, patch_shape, strides=None):
    '''
    Divide numpy image into non-overlapped patches.
    Image tensor axes are arranged in form of [H(eight) W(idth) C(hannel)].
    '''
    if isinstance(patch_shape, int):
        patch_shape = [patch_shape] * 2
    strides = strides or patch_shape
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if len(img.shape) == 2:
        n_channel = 1
    else:
        n_channel = img.shape[-1]
    unit = img.strides[-1]
    cropped_shape = list(img.shape)
    cropped_shape[0] -= cropped_shape[0] % patch_shape[0]
    cropped_shape[1] -= cropped_shape[1] % patch_shape[1]
    # Draw 3D graph of the data, calculate step of jump.
    new_strides = (
        unit*n_channel*img.shape[1]*patch_shape[0], 
        unit*n_channel*patch_shape[1], 
        unit*n_channel*img.shape[1], 
        unit*n_channel, 
        unit,
    )
    new_shape = (
        cropped_shape[0] // patch_shape[0],
        cropped_shape[1] // patch_shape[1],
        patch_shape[0],
        patch_shape[1],
        n_channel,
    )
    return as_strided(img, shape=new_shape, strides=new_strides)

