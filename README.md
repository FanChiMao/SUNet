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
    <td> <img src = "https://imgur.com/dOjxV93" width="400"> </td>
    <td> <img src = "https://i.ibb.co/W0yk5hn/MSC.png" width="400"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Branch U-Net architecture</b></p></td>
    <td><p align="center"> <b>Mixed Skip Connection (MSC)</b></p></td>
  </tr>
</table>

