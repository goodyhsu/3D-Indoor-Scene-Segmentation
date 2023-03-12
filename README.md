# DLCV Final Project ( 3D ScanNet200 long-tail segmentation )

# How to run code
## Installation
We provide a simple bash file to install the environment:
```
git clone https://github.com/goodyhsu/3D-Indoor-Scene-Segmentation.git
cd final-project-challenge-2-il0ve4my
source install.sh
```
Cuda-11.3 is required. Modify `install.sh` if different cuda version is used. See [Install](https://github.com/guochengqian/PointNeXt/blob/master/docs/index.md) for details.

## Setup
```shell script=
bash setup.sh $1
```
* **$1** is the path to the folder of dataset (e.g. \*/dataset/)
    which contains training data folder and testing data folder
    (e.g. \*/dataset/train/\*.ply, \*/dataset/test/\*.ply)

That is, a valid dataset is as below:
```
├── $1
│   ├── train
│   │   ├── *.ply
│   ├── test
│   │   ├── *.ply
```

After running the script, the directory tree containing preprocessed data will look like this (it will be saved to the current working directory):

```
├── dataset_preprocessed
│   ├── train
│   │   ├── *.pth
│   ├── val
│   │   ├── *.pth
│   ├── test
│   │   ├── *.pth
```

    
## Train
```shell script=
bash train.sh
```
* For all the training configurations, please refer to the config file. The file is located at `PointMetaBase/cfgs/scannet_{loss}`, where `loss` can be replaced by either `CE` (cross entropy) or `Focal`, depending on the training scheme that is being used.
* The model checkpoints will be saved to `log/scannet_{loss}` by default.
## Inference
```shell script=
bash inference.sh $1
```
* **$1** is the path to the output directory (e.g. \*/output)
* The script uses checkpoint `best.pth` by default, modify the path if any other checkpoint is to be used. 






## Acknowledgment
This repository is built by reusing codes of [PointNeXt](https://github.com/guochengqian/PointNeXt) and [PointMetaBase](https://github.com/linhaojia13/PointMetaBase). 


## Reference
<a id='1'>[1]</a> 
**PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies**
```tex
@InProceedings{qian2022pointnext,
  title   = {PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies},
  author  = {Qian, Guocheng and Li, Yuchen and Peng, Houwen and Mai, Jinjie and Hammoud, Hasan and Elhoseiny, Mohamed and Ghanem, Bernard},
  booktitle=Advances in Neural Information Processing Systems (NeurIPS),
  year    = {2022},
}
```
<a id='1'>[2]</a> 
**Meta Architecure for Point Cloud Analysis**
```tex
@Article{lin2022meta,
    title={Meta Architecure for Point Cloud Analysis},
    author={Haojia Lin and Xiawu Zheng and Lijiang Li and Fei Chao and Shanshan Wang and Yan Wang and Yonghong Tian and Rongrong Ji},
    journal = {arXiv:2211.14462},
    year={2022},
}
```
