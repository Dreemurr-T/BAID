## Official repository for CVPR2023 paper: "Towards Artistic Image Aesthetics Assessment: a Large-scale Dataset and a New Method"

### Dataset

- Clone the repository
```
git clone https://github.com/Dreemurr-T/BAID.git
cd BAID/
```

- Install the necessary dependencies using:
```
pip install pandas
pip install tqdm
```
- Download the dataset using:
```
python downloading_script/download.py
```
The images will be saved to `images/` folder.

Since it might be slow when downloading the images, we provide alternatives to obtain the dataset:

- Baidu Netdisk: [Link](https://pan.baidu.com/s/19pxr19neJ6Pmd0B6A_u55Q), Code: 9y91
- Google Drive: Coming soon

Ground-truth labels of the dataset can be found in the `dataset` folder.

### Code
#### Requirements

- Python >= 3.8
- Pytorch >= 1.12.0
- Torchvision >= 0.13.0

Other dependencies can be installed with:
```
pip install -r requirements.txt
```

#### Pretraining
- Download the BAID dataset and place the images in the `images/` folder
- Preprocess the data using:
```
python pretraining_utils/pretrain_mani.py
```

- Pretrain the ResNet50 backbone using:
```
python pretraining.py
```
The whole pretraining process takes about 2 days on a single RTX3090. We provide our pretrained weights at [Drive](https://drive.google.com/file/d/13aPiVT4xyu2w5VUwt6vDMJY1n2hXHKWc/view?usp=drive_link).

#### Training

For training on BAID, use:
```
python train.py
```
Checkpoints will be save to `checkpoint/SAAN` folder.

#### Testing

For testing on BAID, download the pretrained weights from [Drive](https://drive.google.com/file/d/1e2XPZjW92HFCUErNHzbmUtk204miAklS/view?usp=drive_link), place the checkpoint in `checkpoint/BAID`

Then use:
```
python test.py
```

### License
The dataset is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)

### Acknowledgement
The code borrowed from [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) and [Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch).

### Citation
If you find our work useful, please cite our work as:
```bibtex
@InProceedings{Yi_2023_CVPR,
    author    = {Yi, Ran and Tian, Haoyuan and Gu, Zhihao and Lai, Yu-Kun and Rosin, Paul L.},
    title     = {Towards Artistic Image Aesthetics Assessment: A Large-Scale Dataset and a New Method},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {22388-22397}
}
```