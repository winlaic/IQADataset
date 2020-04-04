# IQADataset

General dataset for Image Quality Assessment on PyTorch. 



### Introduction

This Python 3 library inherites `torch.utils.data.Dataset` . 

Directly use it in cooperated with `torch.utils.data.Dataloader`.



### Basic Usage

It is extremely easy to use this dataset class. 

Just extract the dataset downloaded without any additional operation and pass the path to the class constructor.

For example:

```python3
from IQADataset import LIVE2016

if __name__ == '__main__':
    dataset = LIVE2016('/path/to/live2016')
    dataset[0] # -> (img, label, distoration_type)
```

`img` is `torch.Tensor` in "CHW" dimention order.

`label` is MOS or DMOS index.



The whole dataset is automaticly separated. 

Training to Evaluating ratio is 8:2. 

`dataset.train()` must be called before training and `dataset.eval()` before evaluating.

Different parts of dataset will be provided according to that.

Call `dataset.all()` to let the dataset provide all images.



### Options

- If you prefer preloading the whole dataset into memory, specify `using_data_pack=True`.

- If you need reference image meanwhile, specify `require_ref=True`.
- You can specify `crop_shape` to apply random crop in train mode.
- Actually, the dataset implements 5-fold strategy. You can specify `n_fold` to reset it.
- Use `dataset.set_i_fold(i)` to specify which part is used for evaluting. Others are used for training.



