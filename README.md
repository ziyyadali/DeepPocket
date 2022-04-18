# PocketFinder

## Introduction
Deep Pocket is an ongoing collaboration with the Gunning Group at the University of Toronto Mississauga. The Gunning Group is a team consisting primarily of medicinal chemists who work on organic synthesis for drug discovery.

The research at the Gunning Group is targeted at battling some of the most aggresive forms of cancer.

## Model Description
We are building a transformer model to predict ligand-binding sites in a protein. Our input is a matrix containing the atomic coordinates represented as a matrix of shape (3, number of atoms). Our output from the model is a singular atomic coordinate in the form of a matrix of shape (3,1). 

## Model Figure
![image](https://user-images.githubusercontent.com/59152943/163754132-c76376a2-4a91-423a-8915-a1738d792426.png)

## Model Parameters
Our model contains a total of $TBD$ parameters. 
-	$d_model$ = 100
-	$max_coords$ = 10000
The multi-head attention contains 4 linear connections. Three of the linear connections correspond to Q, K, and V. The forth linear connection occurs after concatenating the $h$ attention heads. Hence, the number of trainable parameters in the multi-head attention is $4 * ((__a__, __b__) + __a___) = ______$

The encoder portion of the transformer contains $s$ stacks and also contains a Feed Forward layer. The Feed Forward layer is a single-layer perceptron with $(__a__, ___b__) + __a__$ trainable parameters. With the addition of two normalization layers, we would have $2 * (__a__ + __a__)$ parameters. 

The total number in the Encoder portion totals to $s * ((4 * ((____, ___) + ____)) + 2 * (____ + ____) + ((____, ____) + ____))$

The transformer decoder consists of a two multi-head attention layers, three normalization layers and a feed forward layer. The decoder consists of $n$ stacks. The total comes to $n * (2 * (4 * ((___, ___) + ___)) + 3 * (___ + ___) + ((___, ___) + ___))$

The output of the decoder then goes through a linear layer which consists of (___, ___) trainable parameters.


## Model Examples


## Data Source
The data was collected from the Protein data Bank (https://www.rcsb.org/), a database including the atomic coordinates of every protein whose structure has been solved.  

## Data Summary

## Data Transformation and Split
After downloading the .pdb files for $911183$ proteins, a script gathered each coordinate and saved them in a nested list consisting of three separate lists, corresponding the $x, y$ and $z$ coordinates. The list was then written to a pickle (.pkl) file.

Atomic coordinates in the .pdb files look like: 



After running the script, each protein had a .pkl file containing a list:
[[$x_1, x_2, x_3, …$], 
[$y_1, y_2, y_3, …$],
[$z_1, z_2, z_3, …$]]

Since our model is overfitted to a small dataset, we have used ___ proteins to train, ___ for validation, and ___ for testing. 

In the future, we intend to use 100,000 proteins for the validation set, 50,000 for the test set, and the remaining for training.

## Training Curve

## Hyperparameter Tuning

## Quantitative Measures
To measure to accuracy of our model, the predicted atom will need be a distance of ___ angstroms from the ground truth. Distance will be calculated by finding the Euclidean distance between the predicted and known atom, $\sqrt((x_1 – x_2)^2 + (y_1 – y_2)^2 + (z_1 – z_2)^2)$. This distance will be used to determine if the model is actually learning. 

## Results

## Ethical Consideration

## Authors

