# CS598 Deep Learning For Healthcare Project

**Paper Title:** CNN-DDI: a learning-based method for predicting drug–drug interactions using convolution neural networks

**Team ID:** 31 

**Team Members:**
- Avinash Baldeo (@abaldeo2)
- Jinfeng Wu (@jinfeng4)
- Hao Zhang (@haoz18)


## For TA's :point_down: 
1. [Final notebook](https://github.com/abaldeo/CS598_DLH_Project/blob/CNN_DDI/DL4H_Team_31_Final.ipynb)
2. [Final Google Drive Folder](https://drive.google.com/drive/folders/1wSjRXvV91kcJbnOt9nDdHI_8DM-34Gr3?usp=drive_link)
3. [Draft notebook](https://github.com/abaldeo/CS598_DLH_Project/blob/CNN_DDI/DL4H_Team_31_Draft.ipynb)
4. [Draft Google Drive folder](https://drive.google.com/drive/folders/1ln1ga9J7XzwAnAikKXS-ejXcPLe27jgI?usp=drive_link)

## Introduction
The paper we have selected to reproduce is "CNN-DDI: a learning-based method for predicting drug–drug interactions using convolution neural networks" [1]. This research relates to the issue of drug-drug interactions (DDIs) in pharmaceuticals development. Antagonistic DDIs are reactions between two or more drugs that may lead to adverse effects that diminish the efficacy of the drugs involved. Since these drugs are expensive to develop it is important to be able to predict DDIs based on properties of drugs. Knowing if two drugs interact is also useful since drugs similar to either of the two are more likely to interact and cause the same effect.

## CNN-DDI
CNN-DDI predict DDIs by learning from a chosen combination drug features such as categories, targets, pathways, and enzymes. It builds on a previous work “A multimodal deep learning framework for predicting drug-drug interaction events” [2], which uses a DNN model along with four drug features (Target, Enzyme, Pathways and Substructure) to predict DDIs. In the CNN-DDI model, a feature selection framework is constructed to select the best combination of drug features, which is stated to be the Target, Enzyme, Pathways and Category. 

![Table 5.png](https://drive.google.com/uc?export=view&id=1t5hHfM85nWaG4Y6DpIjblb1yuwBSNS84)

## Reproduction Steps
To reproduce the results from the paper, the following steps were performd:
1. Downloaded the dataset collected in DDIMDL repo and added the category data from DrugBank website. 
2. Performed feature extraction to construct feature vectors for target,pathway,enzyme, and category.
3. Performed similarity calculation and created drug similarity matrices using Jaccard similarity (also did for Cosine and Gausine similarity)
4. Implemented the CNN-DDI model as described in the paper, including the convolutional layers, residual block, and activation functions.
5. Trained the CNN-DDI model using the dataset collected and hyperparameter settings found from the CNN-DDI & DDIMDL papers.
6. Evaluated the result using following metrics from the paper:Accuracy, F1-score, Recall, micro-averaged AUPR and micro-averaged AUC 
7. Compared our results with those reported in the CNN-DDI paper.

## Dataset
The primary dataset used by CNN-DDI is from the DDIMDL Github repository (https://github.com/YifanDengWHU/DDIMDL). The DDIMDL paper classifies  DDIs’ events into 65 types and includes 572 drugs with more than 70,000 associated events. The data was originally collected from the DrugBank website (https://go.drugbank.com/)  using a web scraper and then processed and stored into a SQLite database (event.db). To utilize this dataset for CNN-DDI, we had extract the 1622 category types for the drugs in the DDIMDL database from the Drugbank and store it. 

Event.db contains the data we compiled from [DrugBank](https://www.drugbank.ca/) 5.1.3 verision. It has 4 tables:  
**1.drug** contains 572 kinds of drugs and their features.  
**2.event** contains the 37264 DDIs between the 572 kinds of drugs.  
**3.extraction** is the process result of *NLPProcess*. Each interaction is transformed to a tuple: *{mechanism, action, drugA, drugB}*  
**4.event_number** lists the kinds of DDI events and their occurence frequency.  

## Hypothesis Tested
The primary hypothesis tested was that the CNN-DDI model, which utilizes a feature selection framework and a novel CNN architecture, can accurately predict drug-drug interactions and outperform other the models mentioned in the paper (Random forest, Logistic Regression, K-nearest neighbor, Gradient boosted Decision Tree, & DDIMDL). This hypothesis was be tested by training the CNN-DDI model according to the approach mentioned in the paper and evaluating on the collected dataset with the same/inferred settings. Afterwards, we compared the results with table 3 and 4 from the paper to our results. 

## Ablations
To understand the contribution of different components to the model's performance, we performed both feature and model ablation study. 
For feature ablation, we evaluated the model's performance by individually removing drug categories, targets, pathways, and enzyme features. This was done in the CNN-DDI paper and we did the same to allow us to verify the claim made that the drug category is an effective predictor for DDIs. 
For model ablation, we assessed the impact of removing the residual block on the model performance as well as tried using different loss functions (specifically, KL-Divergence and Cosine Similarity).

## Computation Requirements
The CNN-DDI model has approximately 39 million parameters. A single forward pass requires close to 332 million operations as shown below. To run the full cross-validation training and evaluation, it requires a modern, high performance GPU such as Google Colab T4 or GeForce RTX 2080 Ti with at least 11 GBs of memory. It takes on average 60-70 minutes to run 5-fold cross-validation. For each fold, the number of trials run is equal to the number of features. For each trial the number of training epochs is 100. However, since the training process is using the early-stopping strategy (automatically stops the training if no improvement is observed in 10 epochs), the actual number of epochs is usually between 12 and 20.

```
Layers         # operations for a single forward pass
conv1          439,296
conv2          28,114,944
conv3_1        56,229,888
conv3_2        56,229,888
conv4          112,459,776
residual       73,216
fc1            78,194,688
fc2            34,840

Total Operations: 331,776,536
```

## Evaluation
Simply run *CNN_DDI_final.py*, the train-test procedure will start.
 ![Figure1.png](https://drive.google.com/uc?export=view&id=1sWcY2HtiPriRFlBcXqLRakNK73xzjSHg)

The function *prepare* will calculate the similarity between each drug based on their features.  
The function *cross_validation* will take the feature matrix as input to perform 5-CV and calculate metrics. Two csv files will be generated. For example, *pathway+target+enzyme+category_all_CNN_DDI.csv* and *pathway+target+enzyme+category_each_CNN_DDI.csv*. The first file evaluates the method's overall performance while the other evaluates the method's performance on each event. The meaning of the metrics can be seen in array *result_all* and *result_eve* of *CNN_DDI_final.py*.

## Usage
*Example Usage*
```
    python CNN_DDI_final.py -f pathway target enzyme category -c CNN_DDI
```
-f *featureList*: A selection of features to be used in CNN_DDI. The optional features are smile(substructure),target,enzyme, pathway, and category of the drugs. It defaults to pathway,target and enzyme, and category.
-c *classifier*: A selection of prediction method to be used. The optional methods are CNN_DDI, DDIMDL, RF, KNN and LR. It defaults to CNN_DDI.  
-p *NLPProcess*: The choices are *read* and *process*. It means reading the processed result from database directly or processing the raw data again with *NLPProcess.py*. It defaults to *read*. In order to use *NLPProcess.py*, you need to install StanfordNLP package:
```
    pip install stanfordnlp
```
And you need to download english package for StanforNLP:
```
    import stanfordnlp
    stanfordnlp.download('en')
```

## Requirement
Most of the code in this notebook is taken from the github repo for the paper "A multimodal deep learning framework for predicting drug-drug interaction events" [2]. This code was written for Python 3.7 and had the following dependencies.

- numpy (==1.18.1)
- Keras (==2.2.4)
- pandas (==1.0.1)
- scikit-learn (==0.21.2)
- tensorflow (==1.15)
  
The code for this notebook is updated to be run in google colab environment (Python 3.10) and is tested with the following package versions:

- numpy (==1.25.2)
- pandas (==2.0.3)
- scikit-learn (==1.2.2)
- tensorflow (==2.15.0)
- tqdm (==4.66.2)
- psutil (==5.9.5)
- gdown (==4.7.3)

Use the following command to install all dependencies.
```
    # pip install requirement.txt
    pip install numpy==1.25.2
    pip install pandas==2.0.3
    pip install scikit-learn==1.2.2
    pip install tensorflow==2.15.0
    pip install tqdm==4.66.2
    pip install psutil==5.9.5
    pip install gdown==4.7.3
```

## Reproduction Commands
*To reproduce our results, the following commands can be used*
```
#For table 1
python CNN_DDI_final.py
python CNN_DDI_final.py -s Cosine
python CNN_DDI_final.py -s Gaussian
#For table 2
python CNN_DDI_final.py -f target
python CNN_DDI_final.py -f pathway
python CNN_DDI_final.py -f enzyme
python CNN_DDI_final.py -f category
python CNN_DDI_final.py -f pathway target
python CNN_DDI_final.py -f target enzyme
python CNN_DDI_final.py -f target category
python CNN_DDI_final.py -f pathway enzyme
python CNN_DDI_final.py -f pathway category
python CNN_DDI_final.py -f enzyme category
python CNN_DDI_final.py -f pathway target enzyme
python CNN_DDI_final.py -f pathway target category
python CNN_DDI_final.py -f target enzyme category
python CNN_DDI_final.py -f target enzyme category
python CNN_DDI_final.py -f pathway enzyme category
# For table 3
python CNN_DDI_final.py -c RF
python CNN_DDI_final.py -c KNN
python CNN_DDI_final.py -c LR
python CNN_DDI_final.py -c GBDT
# For table 4
python CNN_DDI_final.py -c DDIMDL
# For ablation
python CNN_DDI_final.py -lf kl_divergence
python CNN_DDI_final.py -lf cosine_similarity
```
## Reproduction  Results

![Table 1 R.png](https://drive.google.com/uc?export=view&id=1-aOyN_28loBjH-WiWbjWIC3qgMefPcGH)
![Table 2 R.png](https://drive.google.com/uc?export=view&id=1fbC8dhaDiqN9M4squMcifsFk3jOQcfXn)
![Table 3 R.png](https://drive.google.com/uc?export=view&id=1lFJ5utmt1I12jkLjRmChu4-SDRetxitv)
![Table 4 R.png](https://drive.google.com/uc?export=view&id=1u91bILN3SHaaPubSoQx7aLKGrREt_7kj)
![Ablation Plan.png](https://drive.google.com/uc?export=view&id=1qd_X56IjHvUXGo7PV0Zyl8hKo6nQJjHy)
![Precision-Recall Curves.png](https://drive.google.com/uc?export=view&id=1I7QRRgZzvnUzBKbrPnoqEuKyKUHZ-vEm)
![AUPR and AUC.png](https://drive.google.com/uc?export=view&id=1HPqLHgvh9t0SC3S6eg_rFhzKx44lDvUA)

## Citation  
```
@article{zhang2022cnn,
  title={CNN-DDI: a learning-based method for predicting drug–drug interactions using convolution neural networks},
  author={Zhang, C. and Lu, Y. and Zang, T.},
  journal={BMC Bioinformatics},
  publisher={BioMed Central},
  doi={10.1186/s12859-022-04612-2},
  url={https://doi.org/10.1186/s12859-022-04612-2}
}

@article{deng2020multimodal,
  title={A multimodal deep learning framework for predicting drug-drug interaction events},
  author={Deng, Yifan and Xu, Xinran and Qiu, Yang and Xia, Jingbo and Zhang, Wen and Liu, Shichao},
  journal={Bioinformatics},
  doi={10.1093/bioinformatics/btaa501},
  url={https://doi.org/10.1093/bioinformatics/btaa501}
}
```
