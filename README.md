# Face Verification Using Convolutional Neural Networks

This work is a part of the second homework assignment for introduction to deep learning(CMU-11785) at [Class Link] (https://deeplearning.cs.cmu.edu/F20/index.html). 
In this challenge, will use Convolutional Neural Networks (CNNs) to design an end-to-end system for face verification.
## DATA 
Data can be download at [data link] (https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2)


## DEPENDENCIES

* python 3.6[python package](https://www.python.org/downloads/)
* torch [pytorch package] (https://github.com/pytorch/pytorch)
* numpy [numpy package] (https://numpy.org/install/)
* matplotlib [module link](https://matplotlib.org/) 
* torchvision [model link](https://pypi.org/project/torchvision/0.1.8/)
* sklearn   [package link](https://scikit-learn.org/stable/install.html)
* pillow    [package link](https://pillow.readthedocs.io/en/stable/installation.html)
* pandas    [package link] (https://pandas.pydata.org/docs/getting_started/index.html)

## MODEL ARCHITECTURE

A custom Residual Network is designed that takes input of (batch_size, channel_size, Number of pixels in X, Number of Pixels in Y) and 
outputs a Classification vector of size 4000 with embedding vector of size 2048


## DIRECTORY STRUCTURE

some directories are currently empty (i.e, saved_model, output, data). only data needs to be filled and the rest will be created automically if they do not exit. 
```
hw1_p2
|
|	README.txt
|	submission.csv
|   Classification.ipynb 
|   main.ipynb
|   submission.ipyn
|	
|__saved_model_with_centloss            # currently empty
|	.pt 
|
|__output                               # currently empty
|	.PNG 
|
|__data
|__classification_data
    |_train_data
    |_val_data
|__ verification_data
|__ verification_pairs_test.txt
|__ verification_pairs_val.txt

```
# Note: except the data directory, other directories will be created itself and does not have to be created!!!

## HYPER-PARAMATERS 

*   train_batch_size = 128                      # input batch size for training')
*   test_batch_size  = 64                       # input batch size for training')
*   epochs           = 10                       # number of epochs for training
*   base_lr          = 5.52e-03                 # learning rate for a single GPU
*   lr_cent          = 0.5                      # learning rate for center Loss
*   weight_cent      = 0.15                     # Weight of the Center Loss
*   wd               = 5.0e-04                  # weight decay
*   num_workers      = 4                        # number of worksers for GPU
*   momentum         = 0.9                      # SGD momentum
*   embedding_dim    = 512                      # embedding dimension for images
*   hidden_layers    = [1,1,1,1]                # ResNET hidden Layers

## Optimzers

* SGD with Nesterov


## Learning Rate Scheduler 
*   ReduceLROnPlateau   # parameters can be found on main.ipyn or Classification.ipyn


### TRAINING
* Adjust the directories to your own location
* make sure to have kaggle.json in your directory 
* run the cells 

### Training with Cross Entropy Loss
* run Classification.ipyn

### Training with Center Loss
* run the main.ipyn

### Creating the submission file 
* run the submission.ipyn


### TESTING

* After training is done, pick the model with the best AUC score and evaluate 
    the test data
* I have started training and along the way tuned my parameters without initializing the parameters 
    everytime I started Training. I found this way, the model was initializing from a way better initial 
    point than regular initialization by Kaiming or Xavier. 

## Questions?
shamsbasir@gmail.com

