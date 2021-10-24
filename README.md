# SUNet: Swin Transformer with UNet for Image Denoising (ISCAS 2022)  

## [Chi-Mao Fan](https://github.com/FanChiMao), Tsung-Jung Liu, Kuan-Hsien Liu  

**Paper**:  

**Video Presentation**:  

**Presentation Slides**:  

***
> Abstract : Image restoration is a challenging ill-posed problem
which also has been a long-standing issue. In the past few years
ago, the convolution neural networks (CNN) almost dominated
the compute vision and had achieved considerable success in different
level of vision tasks including image restoration. However,
the Swin Transformer-based model also shows impressive performance,
even suppress the CNN-based methods to become the
state-of-the-art on high-level vision tasks recently. In this paper,
we proposed a restoration model called SUNet which uses the
Swin Transformer layer as our basic blocks and apply to U-Net
architecture for image denoising.

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
To test the pre-trained models of denoising on your own images, run
```
python demo.py --input_dir images_folder_path --result_dir save_images_here --weights path_to_models
```
Here is an example to perform Deraindrop:
```
python demo.py --input_dir './demo_samples/' --result_dir './demo_results' --weights './pretrained_model/denoising_model.pth'
```

## Result  

<img src = "https://i.imgur.com/golsiWN.png" width="800">  

## Visual Comparison  

<img src = "https://i.imgur.com/UeeOO0M.png" width="800">  

<img src = "https://i.imgur.com/YavgU0r.png" width="800">  



## Citation  
If you use SUNet, please consider citing:  
```
@inproceedings{,
    title={},
    author={Chi-Mao Fan, Tsung-Jung Liu, Kuan-Hsien Liu},
    booktitle={ISCAS},
    year={2022}
}
```
