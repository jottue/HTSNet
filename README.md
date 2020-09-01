# HTSNet

# Handwritten Text Separation via End-to-End Learning of Convolutional Neural Network

Junho Jo, Hyung Il Koo, Jae Woong Soh, Nam Ik Cho

[Article Page](https://link.springer.com/article/10.1007/s11042-020-09624-9)

## Environments
- python 3.6
- scipy
- opencv
- numpy
- tqdm


## Abstract

We present a method that separates handwritten and machine-printed components that are mixed and overlapped in documents. Many conventional methods addressed this problem by extracting connected components (CCs) and classifying the extracted CCs into two classes. They were based on the assumption that two types of components are not overlapping
each other, while we are focusing on more challenging and realistic cases where the components are often overlapping or touching each other. For this, we propose a new method that performs pixel-level classification with a convolutional neural network. Unlike conventional neural network methods, our method works in an end-to-end manner and does not require any preprocessing steps (e.g., foreground extraction, handcrafted feature extraction, and so on). For the training of our network, we develop a cross-entropy based loss function to alleviate the class imbalance problem. Regarding the training dataset, although there are some datasets of mixed printed characters and handwritten scripts, most of them do not have many overlapping cases and do not provide pixel-level annotations. Hence, we also propose a data synthesis method that generates realistic pixel-level training samples having many overlappings of printed and handwritten characters.
<br><br>

## Synthesis method

![synthesis_diagram](https://user-images.githubusercontent.com/38808157/90842003-20527e00-e399-11ea-8251-b6b131af7e60.png)



### Prepare datasets
Please download the dataset directly from their website and follow their license agreement.
- [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
- [PRImA Layout Anlysis Dataset](https://www.primaresearch.org/datasets)
- [Scanned Questionnaire Documents](https://drive.google.com/file/d/1-cwOmsBViw5-tJQxcNirWDI90-ZYq1Af/view?usp=sharing)

The structure of the data directory should look like following scripts. If you want to rearrange the structure, you need to edit the file ```data_generation.py``` to reflect the path that you used to store the data.

```
${YOUR_DATA_ROOT}
|-- IAM

|-- Documents
    |-- PRImA
        |-- *.png        
    |-- Questionnaire
        |-- *.png   
```

### Run synthesis code

```
python data_generation.py --data_root ${YOUR_DATA_ROOT} --save_dir ${YOUR_SAVE_DIR} --patch_size 128
```
Running the above command will generate the scribbled document patches in ```${YOUR_SAVE_DIR}``` as shown in following figures:
![typical_example](https://user-images.githubusercontent.com/38808157/91790749-7b635b00-ec4c-11ea-91ec-f442ca9cec34.png)
The first row shows synthesized patches and the second row indicates corresponding pixel-level annotations. Blue, Red and Green denote background, handwritten-text and machine-printed text pixels, respectively. Yellow are overalapping areas.




Updating...
