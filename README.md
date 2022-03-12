# [ISCAS 2022] SUNet: Swin Transformer with UNet for Image Denoising  

## [Chi-Mao Fan](https://github.com/FanChiMao), Tsung-Jung Liu, Kuan-Hsien Liu  

**Paper**: [https://arxiv.org/abs/2202.14009](https://arxiv.org/abs/2202.14009)  
(official version will publish on IEEE after ending of the ISCAS conference on 6/1.)  

**Video Presentation**:  

**Presentation Slides**:  

***
> Abstract : Image restoration is a challenging ill-posed problem
which also has been a long-standing issue. In the past few
years, the convolution neural networks (CNNs) almost dominated
the computer vision and had achieved considerable success
in different levels of vision tasks including image restoration.
However, recently the Swin Transformer-based model also shows
impressive performance, even surpasses the CNN-based methods
to become the state-of-the-art on high-level vision tasks. In this
paper, we proposed a restoration model called SUNet which uses
the Swin Transformer layer as our basic block and then is applied
to UNet architecture for image denoising.


## Network Architecture  

<table>
  <tr>
    <td colspan="2"><img src = "https://i.imgur.com/1UX5j3x.png" alt="CMFNet" width="800"> </td>  
  </tr>
  <tr>
    <td colspan="2"><p align="center"><b>Overall Framework of SUNet</b></p></td>
  </tr>
  
  <tr>
    <td> <img src = "https://imgur.com/lV1CR4H.png" width="400"> </td>
    <td> <img src = "https://imgur.com/dOjxV93.png" width="400"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Swin Transformer Layer</b></p></td>
    <td><p align="center"> <b>Dual up-sample</b></p></td>
  </tr>
</table>

## Quick Run  
You can directly run personal noised images on my space of [HuggingFce](https://huggingface.co/spaces/52Hz/SUNet_AWGN_denoising).  

To test the [pre-trained models](https://drive.google.com/file/d/1ViJgcFlKm1ScEoQH616nV4uqFhkg8J8D/view?usp=sharing) of denoising on your own 256x256 images, run
```
python demo.py --input_dir images_folder_path --result_dir save_images_here --weights path_to_models
```
Here is an example command:
```
python demo.py --input_dir './demo_samples/' --result_dir './demo_results' --weights './pretrained_model/denoising_model.pth'
```
To test the pre-trained models of denoising on your arbitrary resolution images, run
```
python demo_any_resolution.py --input_dir images_folder_path --stride shifted_window_stride --result_dir save_images_here --weights path_to_models
```
SUNset could only handle the fixed size input which the resolution in training phase same as the mostly transformer-based methods because of the attention masks are fixed. If we want to denoise the arbitrary resolution input, the shifted-window method will be applied to avoid border effect. The code of `demo_any_resolution.py` is supported to fix the problem.

## Train  
To train the restoration models of Denoising. You should check the following components:  
- `training.yaml`:  

  ```
    # Training configuration
    GPU: [0,1,2,3]

    VERBOSE: False

    SWINUNET:
      IMG_SIZE: 256
      PATCH_SIZE: 4
      WIN_SIZE: 8
      EMB_DIM: 96
      DEPTH_EN: [8, 8, 8, 8]
      HEAD_NUM: [8, 8, 8, 8]
      MLP_RATIO: 4.0
      QKV_BIAS: True
      QK_SCALE: 8
      DROP_RATE: 0.
      ATTN_DROP_RATE: 0.
      DROP_PATH_RATE: 0.1
      APE: False
      PATCH_NORM: True
      USE_CHECKPOINTS: False
      FINAL_UPSAMPLE: 'Dual up-sample'

    MODEL:
      MODE: 'Denoising'

    # Optimization arguments.
    OPTIM:
      BATCH: 4
      EPOCHS: 500
      # EPOCH_DECAY: [10]
      LR_INITIAL: 2e-4
      LR_MIN: 1e-6
      # BETA1: 0.9

    TRAINING:
      VAL_AFTER_EVERY: 1
      RESUME: False
      TRAIN_PS: 256
      VAL_PS: 256
      TRAIN_DIR: './datasets/Denoising_DIV2K/train'       # path to training data
      VAL_DIR: './datasets/Denoising_DIV2K/test' # path to validation data
      SAVE_DIR: './checkpoints'           # path to save models and images
  ```
- Dataset:  
  The preparation of dataset in more detail, see [datasets/README.md](datasets/README.md).  
  
- Train:  
  If the above path and data are all correctly setting, just simply run:  
  ```
  python train.py
  ```  
## Result  

<img src = "https://i.imgur.com/golsiWN.png" width="800">  

## Visual Comparison  

<img src = "https://i.imgur.com/UeeOO0M.png" width="800">  

<img src = "https://i.imgur.com/YavgU0r.png" width="800">  



## Citation  
If you use SUNet, please consider citing:  
```
@article{fan2022sunet,
  title={SUNet: Swin Transformer UNet for Image Denoising},
  author={Fan, Chi-Mao and Liu, Tsung-Jung and Liu, Kuan-Hsien},
  journal={arXiv preprint arXiv:2202.14009},
  year={2022}
}
```
