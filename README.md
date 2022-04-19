# Deep Pocket: Ligand Binding Site Detection using Transformer Model

## Introduction
Deep Pocket is an ongoing collaboration with the Gunning Lab at the University of Toronto Mississauga. The Gunning Lab is a team consisting primarily of medicinal chemists who work on organic synthesis for drug discovery. Deep Pocket is a framework for ligand binding site detection from protein structures.

## Motivation

The research at the Gunning Lab is targeted at battling some of the most aggresive forms of cancer, including brain, breast, and blood cancer. For chemists to synthesize such inhibitory molecules, they must be able to identify whether a protein has one or more potential binding sites. After successfully identifying at least one binding site, they would elucidate amino acid residues in these sites, and design small molecules to bind. This task is typically done through literature review, however, when encountering new proteins the process may become long and arduous. Researchers, including the team at the Gunning Group, may determine this in a quite rudimentary manner, such as plotting the atomic coordinates on Microsoft Excel as a three-dimensional volume, and scanning for pocket-like structures with the naked eye.

This project is our team's proposal for a deep learning solution to this problem.  In the process, our team envisions being a forerunner to the growing presence of deep learning in the fields of medicinal chemistry and drug discovery for cancer research.

## Model Description
The transformer model is used to predict ligand-binding sites in a protein structure. The input to the model is a tensor containing the atomic coordinates represented as a tensor of shape `(number of proteins, number of atoms, 3)` where the number of proteins is analogous to the batch size. The output from the model is a sigular atomic coordinate per protein in the form of a matrix of shape `(number of proteins, 1, 3)`. 

## Model Figure
Due to time constraints, we were unable to complete a full transformer model based on the paper "Attention Is All You Need" architecture. Currently, our model contains the encoder portion of the complete model. 

![image](https://user-images.githubusercontent.com/59152943/163836491-614a2039-6e95-49a9-93ae-71c4daad9d0a.png)

The input will pass through the embedder which transforms the data to a shape of `(number of proteins, max atoms, 3, d_model)`. The tensor is then sent to the Multi-Head Attention module. 

![image](https://user-images.githubusercontent.com/59152943/163839504-600dd4b2-8e39-4113-8a2d-361e617e763d.png)

The input is passed through three linear layers, Q, K, and V. The tensors are reshaped and the attention scores are calculated and concatenated. The result goes into a linear layer and is outputted with a shape of `(number of proteins, d_model)`. 

The tensor then goes though the FeedForward module that reduces the dimentions result in a tensor of shape `(number of proteins, 3)`.

The following image represents the "Attention Is All You Need" architecture. 
![image](https://user-images.githubusercontent.com/59152943/163833635-a418d5e9-9400-467e-9078-dad7d706147c.png)
 
In the future, we intend to improve upon the model and add more modules to create a complete Transformer. 

## Model Parameters
Our model contains a total of `86,624` parameters. 
-	`d_model = 144` This represents the model dimension
-	`max_coords = 100000` 
The multi-head attention contains 4 linear connections. Three of the linear connections correspond to Q, K, and V. The forth linear connection occurs after concatenating the `h` attention heads. Hence, the number of trainable parameters in the multi-head attention is `4 * ((d_model, d_model) + d_model) = 83,520`

The encoder portion of the transformer contains `s` stacks and also contains a Feed Forward layer. The Feed Forward layer is a multi-layer perceptron with `((d_model, 20) + d_model) + ((20, 3) + 20) = 3,024 + 80 = 3,104` trainable parameters. 

The total number in the Encoder portion, where `s = 1` totals to `s (83,520 + 3,104) = (83,520 + 3,104) = 86,624` trainable parameters.

## Model Evaluation Example
Model performance using 12 proteins in the test set is shown below:
```
Model Accuracy: 0.9166666666666666
Incorrect: 1
Correct: 11
```
Due to the nature of how the data is organized, It was hard to identify the correctly classified or misclassified protein. 

## Data Source
The data was collected from the Protein data Bank (https://www.rcsb.org/), a database including the atomic coordinates of every protein whose structure has been solved.  

## Data Summary

## Data Transformation and Split
After downloading the .pdb files for $911183$ proteins, a python script gathered each coordinate and saved them in a nested list consisting of three separate lists, corresponding to the $x, y$ and $z$ coordinates. The list was then written to a pickle (.pkl) file.

Atomic coordinates in the .pdb files look like: 
```
...
ATOM      1  N   LYS A   2     140.951 118.441 109.053  1.00 67.48           N  
ATOM      2  CA  LYS A   2     140.745 118.741 110.465  1.00 67.16           C  
ATOM      3  C   LYS A   2     141.491 120.006 110.875  1.00 66.83           C  
ATOM      4  O   LYS A   2     141.842 120.828 110.026  1.00 66.58           O  
ATOM      5  CB  LYS A   2     141.189 117.560 111.333  1.00 67.42           C  
ATOM      6  CG  LYS A   2     142.657 117.189 111.180  1.00 67.04           C  
ATOM      7  CD  LYS A   2     143.017 115.992 112.046  1.00 66.78           C  
ATOM      8  CE  LYS A   2     144.516 115.740 112.043  1.00 66.62           C  
ATOM      9  NZ  LYS A   2     145.032 115.462 110.674  1.00 66.31           N1+
ATOM     10  N   GLU A   3     141.720 120.152 112.179  1.00 66.59           N  
ATOM     11  CA  GLU A   3     142.501 121.259 112.728  1.00 66.55           C  
ATOM     12  C   GLU A   3     141.898 122.620 112.365  1.00 65.69           C  
ATOM     13  O   GLU A   3     142.432 123.352 111.532  1.00 65.08           O  
ATOM     14  CB  GLU A   3     143.954 121.163 112.252  1.00 66.91           C  
ATOM     15  CG  GLU A   3     144.924 122.062 112.993  1.00 67.33           C  
ATOM     16  CD  GLU A   3     145.628 123.033 112.073  1.00 67.65           C  
ATOM     17  OE1 GLU A   3     145.549 122.846 110.840  1.00 67.95           O 
...
```

This produces a visual representation of the protein. 

![image](https://user-images.githubusercontent.com/59152943/163837875-0ca9c7f0-fbf1-4b1e-9cfb-08c1578bc27c.png)

After running the script, each protein had a .pkl file containing a list with two elements `[coordinates, ligands]`:
```
[[[x1, y1, z1],   \
[x2, y2, z2],      } coordinates 
[x3, y3, z3],     /
...],
[[x1, y1, z3],     } ligands (can be empty)
...]]
```

The model was trained on the data from 12 protein structures.The training was restricted to the subset of the data mainly due to the training time. 

In the future, we intend to use 100,000 proteins for the validation set, 50,000 for the test set, and the remaining for training the model.

## Training Curve
![image](https://user-images.githubusercontent.com/59152943/163754132-c76376a2-4a91-423a-8915-a1738d792426.png)

The training progress output for Trial 3 (mentioned in HyperParameter Tuning):
```
Epoch 0. Iter 0. [Train Acc 100%, Loss 75.694145]
Epoch 12. Iter 50. [Train Acc 100%, Loss 47.215687]
Epoch 25. Iter 100. [Train Acc 100%, Loss 33.217171]
Epoch 37. Iter 150. [Train Acc 100%, Loss 38.528770]
Time of epoch 50 is 1.176262
Epoch 50. Iter 200. [Train Acc 100%, Loss 32.316666]
Epoch 62. Iter 250. [Train Acc 100%, Loss 29.795990]
Epoch 75. Iter 300. [Train Acc 100%, Loss 31.131182]
Epoch 87. Iter 350. [Train Acc 100%, Loss 18.106314]
Time of epoch 100 is 1.160259
Epoch 100. Iter 400. [Train Acc 100%, Loss 18.442919]
Epoch 112. Iter 450. [Train Acc 100%, Loss 35.407032]
Epoch 125. Iter 500. [Train Acc 100%, Loss 24.975988]
Epoch 137. Iter 550. [Train Acc 100%, Loss 16.319950]
Time of epoch 150 is 1.173262
Epoch 150. Iter 600. [Train Acc 100%, Loss 20.180580]
Epoch 162. Iter 650. [Train Acc 100%, Loss 26.963339]
Epoch 175. Iter 700. [Train Acc 100%, Loss 20.545383]
Epoch 187. Iter 750. [Train Acc 100%, Loss 16.830923]
Time of epoch 200 is 1.199270
Epoch 200. Iter 800. [Train Acc 100%, Loss 14.469269]
Epoch 212. Iter 850. [Train Acc 100%, Loss 24.579166]
Epoch 225. Iter 900. [Train Acc 100%, Loss 28.672127]
Epoch 237. Iter 950. [Train Acc 100%, Loss 20.857580]
Time of epoch 250 is 1.279285
Epoch 250. Iter 1000. [Train Acc 100%, Loss 28.767866]
Epoch 262. Iter 1050. [Train Acc 100%, Loss 39.256058]
Epoch 275. Iter 1100. [Train Acc 100%, Loss 26.106178]
Epoch 287. Iter 1150. [Train Acc 100%, Loss 10.657035]
Time of epoch 300 is 1.204269
Epoch 300. Iter 1200. [Train Acc 100%, Loss 19.187923]
Epoch 312. Iter 1250. [Train Acc 100%, Loss 24.099941]
Epoch 325. Iter 1300. [Train Acc 100%, Loss 21.364412]
Epoch 337. Iter 1350. [Train Acc 100%, Loss 25.114309]
Time of epoch 350 is 1.120250
Epoch 350. Iter 1400. [Train Acc 100%, Loss 21.217381]
Epoch 362. Iter 1450. [Train Acc 100%, Loss 18.213755]
Epoch 375. Iter 1500. [Train Acc 100%, Loss 24.732666]
Epoch 387. Iter 1550. [Train Acc 100%, Loss 18.474697]
Time of epoch 400 is 1.120250
Epoch 400. Iter 1600. [Train Acc 100%, Loss 6.061907]
Epoch 412. Iter 1650. [Train Acc 100%, Loss 19.984243]
Epoch 425. Iter 1700. [Train Acc 100%, Loss 8.701896]
Epoch 437. Iter 1750. [Train Acc 100%, Loss 21.178425]
Time of epoch 450 is 1.119255
Epoch 450. Iter 1800. [Train Acc 100%, Loss 17.429441]
Epoch 462. Iter 1850. [Train Acc 100%, Loss 18.401318]
Epoch 475. Iter 1900. [Train Acc 100%, Loss 12.048043]
Epoch 487. Iter 1950. [Train Acc 100%, Loss 18.715904]
Time of epoch 500 is 1.120757
```

## Hyperparameter Tuning
We initially used following parameters setting:[^1]
- learning rate: 0.0001
- epochs: 1000
- batch size: 5
- total proteins: 10

Trail 2:
- learning rate: 0.001
- epochs: 500
- batch size: 2
- total proteins: 10

Trial 3:
- learning rate: 0.001
- epochs: 500
- batch size: 3
- total proteins: 12

[^1]: This was trained using a CrossEntropyLoss loss function. This resulted in a very high loss. The other training models used the L1Loss function.  

## Quantitative Measures
The model correctly classified the protein binding if the predicted atom was within a distance of 215 angstroms(21.5 nanometres) from the ground truth. Distance between the predicted and target atom was calculated using the Euclidean distance, `sqrt((x1 – x2)**2 + (y1 – y2)**2 + (z1 – z2)**2)`. This distance was used to quantify the model learning. 

## Results
Due to time constraints and hardware limitations, we were only able to produce an overfitting model trained on 12 proteins. The training model used the following hyperparameter values:
- learning rate: 0.001
- epochs: 500
- batch size: 3
- total proteins: 12
We utilized checkpoint 425 which resulted in a Training Accuracy of 100% and a Loss of 8.701896. 

Based on the current model performance, we believe that the model is not appropriate for deployment. This is mostly because our model is incomplete, so it has relatively large margain of error in the predictions. Once we are able to successfuly incorporate decoder architecture in the model, we can then make use of the expected ligands and create some correlation between the encoder and decoder. The model would also benefit from using more than one stack of the encoder. 

The accuracy function could be altered to better reflect the progress of the model. A decreasing range threshold would also be a viable measurement.  

The test accuracy of the model was 91.67%. However, the accuracy function exagerates the peformance greatly. In the future, we would want to create a more accurate model that gets a high accuracy with a range of 10 angstroms.  

## Ethical Consideration
It is difficult to foresee a serious misuse of our technology, as our model is designed for medicinal purposes. Since our technology is designed primarily for the lab as opposed to clinical practice, in the event of failure to make an accurate prediction, our model would result in lost time and money for the researcher. This, of course, is inevitable in the research process. The only thing that needs to be necessary for the use of our model is that the number of atomic coordinates must be less than or equal to 100000, because that is the max number of coordinates the embedder is designed to handle. In the future, it can be dynamically changed, where the model needs to be retrained.

Furthermore, it is in the realm of possibility for a researcher to maliciously design drugs with our technology, in order to inhibit proteins crucial for healthy, and orderly functioning of the human body.

## Authors
Every team member contributed to the success of the project. More detailed breakdown is presented below.
- Preprocessing: Raiyan and Ahmed created a script to download large amounts of data and found the coordinates of each atom and the ligands. They were then put into lists to use for our model.
- Model Creation: Raiyan, Ziyyad, Ahmed, and Hrithik developed the model together. 
- Model Training: Ziyyad and Raiyan trained and created an overfitting model. They also tested various hyperparameters and ranges for the accuracy function. 
- README writeup: Ziyyad wrote the writeup and tested the model on a subset of the testing data with help from Raiyan and Hrithik . 


