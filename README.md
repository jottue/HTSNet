# HTSNet

# Handwritten Text Separation via End-to-End Learning of Convolutional Neural Network

Junho Jo, Hyung Il Koo, Jae Woong Soh, Nam Ik Cho

[Article](https://link.springer.com/article/10.1007/s11042-020-09624-9)

## Environments
- [Tensorflow 1.13](http://www.tensorflow.org/)
- Python 3.6

## Abstract

We present a method that separates handwritten and machine-printed components that are mixed and overlapped in documents. Many conventional methods addressed this problem by extracting connected components (CCs) and classifying the extracted CCs into two classes. They were based on the assumption that two types of components are not overlapping
each other, while we are focusing on more challenging and realistic cases where the components are often overlapping or touching each other. For this, we propose a new method that performs pixel-level classification with a convolutional neural network. Unlike conventional neural network methods, our method works in an end-to-end manner and does not require any preprocessing steps (e.g., foreground extraction, handcrafted feature extraction, and so on). For the training of our network, we develop a cross-entropy based loss function to alleviate the class imbalance problem. Regarding the training dataset, although there are some datasets of mixed printed characters and handwritten scripts, most of them do not have many overlapping cases and do not provide pixel-level annotations. Hence, we also propose a data synthesis method that generates realistic pixel-level training samples having many overlappings of printed and handwritten characters.
<br><br>

## Synthesis method

### Donwload datasets
Please download the dataset directly from their website and follow their license agreement.
- [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
- [PRImA Layout Anlysis Dataset](https://www.primaresearch.org/datasets)
- [Scanned Questionnaire Documents](https://drive.google.com/file/d/1-cwOmsBViw5-tJQxcNirWDI90-ZYq1Af/view?usp=sharing)


### Synthesis Diagram

![synthesis_diagram](https://user-images.githubusercontent.com/38808157/90842003-20527e00-e399-11ea-8251-b6b131af7e60.png)


Updating...
