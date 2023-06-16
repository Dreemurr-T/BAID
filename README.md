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

Ground-truth labels of the train set can be found in the `train` folder. However, since we are currently holding a competition at [CGI-AIAA 2023](https://codalab.lisn.upsaclay.fr/competitions/12790), we cannot release the ground-truth labels of the test set right now. We apologize for the inconvenience. 

### Code
#### Requirements

- Python >= 3.8
- Pytorch >= 1.12.0
- Torchvision >= 0.13.0

Other dependencies can be installed with:
```
pip install -r requirements.txt
```

#### Training
- Download the BAID dataset and place the images in the `images/` folder
- For training on BAID, use:
```
python train.py
```
Checkpoints will be save to `checkpoint/SAAN` folder.

#### Testing

For testing on BAID, use:
```
python test.py
```
Pretrained models and ground-truth labels of the test set will be released after the [CGI-AIAA 2023](https://codalab.lisn.upsaclay.fr/competitions/12790) challenge.

### License
The dataset is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)

### Acknowledgement
The code borrowed from [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) and [Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch).