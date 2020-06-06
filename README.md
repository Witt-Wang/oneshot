# One-Shot Learning for Long-Tail Visual Relation Detection

This is a PyTorch implementation for [One-Shot Learning for Long-Tail Visual Relation Detection, AAAI2020] This is an **improved version** of the code.

## News
Because of the new coronavirus, we can't go back to school, so our open source work was delayed. We will upload code as soon as possible. If you have any questions, please contact the first author.

## Benchmarking on Visual Genome

|                    |         |VRD-One | |                    | |                    | VG-One |     |               |
|--------------------|:--------:|:--------:|:--------:|:--------:|:-----:|:--------:|:-----:|:-----:|
|                    | PredCls |  | SGCls | | PredCls| | SGCls||

|                    | 5-way 1-shot | 10-way 1-shot | 5-way 1-shot | 10-way 1-shot | 5-way 1-shot | 10-way 1-shot | 5-way 1-shot| 10-way 1-shot|


| Ours(old    | 48.4%        | 33.5%        | 22.3%       | 20.9%        | 56.3%    | 37.5%       | 14.9%     | 13.2%|
| Ours        | 49.9%        | 35.9%        | 25.2%        | 19.5%       | 56.3%    | 40.7%       | 15.2%     | 14.3%|
## Requirements
* Python 3
* Python packages
  * pytorch 1.0
  * cython
  * matplotlib
  * numpy
  * scipy
  * opencv
  * pyyaml
  * packaging
  * tensorboardX
  * tqdm
  * pillow
  * scikit-image
* An NVIDIA GPU and CUDA 8.0 or higher. Some operations only have gpu implementation.


### VG-one
Download it [here]. Unzip it under the data folder. You should see a `vg-one` folder unzipped there. It contains .json annotations that suit the dataloader used in this repo.

### VRD-one
Download it [here]. Unzip it under the data folder. You should see a `vg-one` folder unzipped there. It contains .json annotations that suit the dataloader used in this repo.


## Directory Structure


## Evaluating Pre-trained models
```
python main.py

```
## Citing
If you use this code in your research, please use the following BibTeX entry.
