# HTSNet

# Handwritten Text Separation via End-to-End Learning of Convolutional Neural Network

Junho Jo, Hyung Il Koo, Jae Woong Soh, Nam Ik Cho


## Environments
- Ubuntu 18.04
- [Tensorflow 1.8](http://www.tensorflow.org/)
- CUDA 9.0 & cuDNN 7.1
- Python 3.6

## Abstract

We present a method that separates handwritten and machine-printed components that are mixed and overlapped in documents. Many conventional methods addressed this problem by extracting connected components (CCs) and classifying the extracted CCs into two classes. They were based on the assumption that two types of components are not overlapping
each other, while we are focusing on more challenging and realistic cases where the components are often overlapping or touching each other. For this, we propose a new method that performs pixel-level classification with a convolutional neural network. Unlike conventional neural network methods, our method works in an end-to-end manner and does not require any preprocessing steps (e.g., foreground extraction, handcrafted feature extraction, and so on). For the training of our network, we develop a cross-entropy based loss function to alleviate the class imbalance problem. Regarding the training dataset, although there are some datasets of mixed printed characters and handwritten scripts, most of them do not have many overlapping cases and do not provide pixel-level annotations. Hence, we also propose a data synthesis method that generates realistic pixel-level training samples having many overlappings of printed and handwritten characters.
<br><br>

## Synthesis method

### Used DataSet
Download these datasets following likned url. Some dataset need authentfication for lisence.
- [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
- [PRImA]
- Scanned



### Super-Resolution for Various Kernels

#### [ZSSR (CVPR 2018)] "Zero-Shot" Super-Resolution Using Deep Internal Learning <a href="http://openaccess.thecvf.com/content_cvpr_2018/html/Shocher_Zero-Shot_Super-Resolution_Using_CVPR_2018_paper.html">Link</a> 

#### [IKC (CVPR 2019)] Blind Super-Resolution With Iterative Kernel Correction <a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Gu_Blind_Super-Resolution_With_Iterative_Kernel_Correction_CVPR_2019_paper.html">Link</a> 

### Optimization-based Meta-Learning

#### [MAML (ICML 2017)] Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks <a href="https://arxiv.org/abs/1703.03400">Link</a>

#### [MAML++ (ICLR 2019)] How to train your MAML <a href="https://arxiv.org/abs/1810.09502">Link</a>
<br><br>

## Brief Description of Our Proposed Method

### <u>Illustration of the Overall Scheme</u>

<p align="center"><img src="/figure/synthesis_diagram.png width="700"></p>

During meta-transfer learning, the external dataset is used, where internal learning is done during meta-test time.
From random initial \theta_0, large-scale dataset DIV2K with “bicubic” degradation is exploited to obtain \theta_T.
Then, meta-transfer learning learns a good representation \theta_M for super-resolution tasks with diverse blur kernel scenarios.
In the meta-test phase, self-supervision within a test image is exploited to train the model with corresponding blur kernel.

### <u> Algorithms </u>

<p align="center"><img src="figure/meta-training.png" width="400">&nbsp;&nbsp;<img src="figure/meta-test.png" width="400"></p> 

Left: The algorithm of Meta-Transfer Learning & Right: The algorithm of Meta-Test.

## Experimental Results

**Results on various kernel environments (X2)**

<p align="center"><img src="figure/result.png" width="900"></p>

The results are evaluated with the average PSNR (dB) and SSIM on Y channel of YCbCr colorspace.
<font color="red">Red </font> color denotes the best results and <font color ="blue"> blue </font> denotes the second best.
The number between parantheses of our methods (MZSR) denote the number of gradient updates.

**Results on scaling factor (X4)**

<p align="center"><img src="figure/resultx4.png" width="900"></p>

**Test Input Data**

Degraded Images of Set5, B100, Urban100 on various kernel environments.

[Download](https://drive.google.com/open?id=16L961dGynkraoawKE2XyiCh4pdRS-e4Y)

## Visualized Results

<p align="center"><img src="figure/001.png" width="900"></p>
<br><br>
<p align="center"><img src="figure/002.png" width="900"></p>

## Brief explanation of contents

```
├── GT: Ground-truth images
├── Input: Input LR images
├── Model: Pre-trained models are included (Model Zoo)
    ├──> Directx2: Model for direct subsampling (x2)
    ├──> Multi-scale: Multi-scale model
    ├──> Bicubicx2: Model for bicubic subsampling (x2)
    └──> Directx4: Model for direct subsampling (x4)
├── Pretrained: Pre-trained model (bicubic) for transfer learning.
└── results: Output results are going to be saved here.

Rest codes are for the training and test of MZSR.
```

## Guidelines for Codes

**Requisites should be installed beforehand.**

Clone this repo.
```
git clone http://github.com/JWSoh/MZSR.git
cd MZSR/
```

### Training

Download training dataset [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

#### Generate TFRecord dataset
- Refer to [MainSR](https://www.github.com/JWSoh/MainSR) repo.
- Run generate_TFRecord_MZSR.py

#### Train MZSR
Make sure all configurations in **config.py** are set.

[Options]
```
python main.py --train --gpu [GPU_number] --trial [Trial of your training] --step [Global step]

--train: Flag in order to train.
--gpu: If you have more than one gpu in your computer, the number denotes the index. [Default 0]
--trial: Trial number. Any integer numbers can be used. [Default 0]
--step: Global step. When you resume the training, you need to specify the right global step. [Default 0]
```

### Test

Ready for the input data (low-resolution) and corresponding kernel (kernel.mat file.)

[Options]
```
python main.py --gpu [GPU_number] --inputpath [LR path] --gtpath [HR path] --savepath [SR path]  --kernelpath [kernel.mat path] --model [0/1/2/3] --num [1/10]

--gpu: If you have more than one gpu in your computer, the number designates the index of GPU which is going to be used. [Default 0]
--inputpath: Path of input images [Default: Input/g20/Set5/]
--gtpath: Path of reference images. [Default: GT/Set5/]
--savepath: Path for the output images. [Default: results/Set5]
--kernelpath: Path of the kernel.mat file. [Default: Input/g20/kernel.mat]
--model: [0/1/2/3]
    -> 0: Direct x2
    -> 1: Multi-scale
    -> 2: Bicubic x2
    -> 3: Direct x4
--num: [1/10] The number of adaptation (gradient updates). [Default 1]

```

You may change other minor options in "test.py."
Line 9 to line 17.

The minor options are shown below.
```
self.save_results=True		-> Whether to save results or not.
self.display_iter = 1		-> The interval of information display.
self.noise_level = 0.0		-> You may sometimes add small noise for real-world images.
self.back_projection=False	-> You may also apply back projection algorithm for better results.
self.back_projection_iters=4	-> The number of iteration of back projection.
```

### An example of test codes

```
python main.py --gpu 0 --inputpath Input/g20/Set5/ --gtpath GT/Set5/ --savepath results/Set5 --kernelpath Input/g20/kernel.mat --model 0 --num 1
```

## Citation
```
@article{soh2020meta,
  title={Meta-Transfer Learning for Zero-Shot Super-Resolution},
  author={Soh, Jae Woong and Cho, Sunwoo and Cho, Nam Ik},
  journal={arXiv preprint arXiv:2002.12213},
  year={2020}
}

@inproceedings{soh2020meta,
  title={Meta-Transfer Learning for Zero-Shot Super-Resolution},
  author={Soh, Jae Woong and Cho, Sunwoo and Cho, Nam Ik},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Acknowledgement
Our work and implementations are inspired by and based on
ZSSR [[site](https://github.com/assafshocher/ZSSR)] and MAML [[site](https://github.com/cbfinn/maml)].
