# SC-Net
# Weakly Supervised Medical Image Segmentation via Superpixel-guided Scribble Walking and Class-wise Contrastive Regularization

Pytorch implementation of our Weakly Supervised Medical Image Segmentation via Superpixel-guided Scribble Walking and Class-wise Contrastive Regularization. <br/>

## Paper
[Weakly Supervised Medical Image Segmentation via Superpixel-guided Scribble Walking and Class-wise Contrastive Regularization](https://github.com/Lemonzhoumeng/SC-Net) MICCAI 2023
<p align="center">
  <img src="figure/framework.png">
</p>

## Installation
* Install Pytorch 0.4.1 and CUDA 9.0 (Note that the results reported in the paper are obtained by running the code on this Pytorch version. As raised by the issue, using higher version of Pytorch may seem to have a performance decrease on optic cup segmentation.)
* Clone this repo
```
git clone https://github.com/Lemonzhoumeng/SC-Net
cd SC-Net
```

## Train
* Download datasets from [here](https://drive.google.com/file/d/1B7ArHRBjt2Dx29a3A6X_lGhD0vDVr3sy/view).
* Download source domain model from [here](https://www.dropbox.com/s/qygkmpm6ez6bojd/source_model.pth.tar?dl=0) or specify the data path in `./train_source.py` and then train `./train_source.py`.
* Save source domain model into folder `./logs/source`.
* Download generated pseudo labels from [here](https://www.dropbox.com/s/opuz9pt78ng1yds/pseudolabel.zip?dl=0) or specify the model path and data path in `./generate_pseudo.py` and then train `./generate_pseudo.py`.
* Save generated pseudo labels into folder `./generate_pseudo`.
* Run `./train_target.py` to start the target domain training process.

## Acknowledgement
The code for source domain training is modified from [WSL4MIS]([https://github.com/HiLab-git/WSL4MIS]). 


## Note
* Contact: Meng Zhou (1155156866@link.cuhk.edu.hk)
