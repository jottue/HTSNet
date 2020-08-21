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
Download these datasets following linked URL. Some dataset need authentication for lisence.
- [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
- [PRImA Layout Anlysis Dataset](https://www.primaresearch.org/datasets)
- [Scanned questionnaire form documents](https://drive.google.com/file/d/1jhS52PuD_gNa-BVHpH3j7hZdynw0ZSP5/view?usp=sharing)

Updating...
