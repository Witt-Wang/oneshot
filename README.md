# One-Shot Learning for Long-Tail Visual Relation Detection

This is a PyTorch implementation for [One-Shot Learning for Long-Tail Visual Relation Detection] This is an **improved version** of the code.

## News
Because of the new coronavirus, we can't go back to school, so our open source work was delayed. We will upload code as soon as possible. If you have any questions, please contact the first author.

## More

- [ ] code optimization
- [ ] end to end

## Benchmarking on VG-one and VRD-one
VRD-one
|                    | PredCls 5-way 1-shot | PredCls 10-way 1-shot |SGCls 5-way 1-shot | SGCls 10-way 1-shot | 
|--------------------|:--------:|:--------:|:--------:|:--------:|
| Ours(old    | 48.4%        | 33.5%        | 22.3%       | 20.9%        | 
| Ours        | 49.9%        | 35.9%        | 25.2%        | 19.5%       | 

VG-one
|                    | PredCls 5-way 1-shot | PredCls 10-way 1-shot | SGCls 5-way 1-shot| SGCls 10-way 1-shot|
|--------------------|:--------:|:--------:|:--------:|:--------:|
| Ours(old    | 56.3%    | 37.5%       | 14.9%     | 13.2%|
| Ours        | 56.3%    | 40.7%       | 15.2%     | 14.3%|


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
Download it [here](https://1drv.ms/u/s!AusVrwCGXciOlBI6p1vtLbVVAMD-?e=6MZplh). Unzip it under the data folder. You should see a `vg-one` folder unzipped there. It contains .json annotations that suit the dataloader used in this repo.

### VRD-one
Download it [here](https://1drv.ms/u/s!AusVrwCGXciOlBGt-JRl7Ihk5Ame?e=btwGDf). Unzip it under the data folder. You should see a `vg-one` folder unzipped there. It contains .json annotations that suit the dataloader used in this repo.


## Directory Structure

## Getting Started
### Prepare datasets
Our model is based on Faster RCNN, you need to use Faster RCNN model to extract image features, and put them in the `$oneshot/data`.
### Train a model
```
python main.py
```
## Citing
If you use this code in your research, please use the following BibTeX entry.
