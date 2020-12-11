# CZ/CE4042-NN&DL-Group Project

## Running Directory
The source directory has all ipython scripts for the assignment. 
```
$ cd source
```

## Contents:

* gender_preprocessing.ipynb : for preprocessing of dataset and generate pkl files
 
* from_scratch : store the experiments for hyperparameters tuning and training of their optimal models
* 
  * baseline_gender_fold0.ipynb
  * depth_experiments_fold0.ipynb
  * dropout_gender_fold0.ipynb
  * optimal_baseline_gender.ipynb
  * optimal_dropout_gender.ipynb
  * optimal_width_expreiment_gender.ipynb
  * optimizer_gender_fold0.ipynb
  * width_experiment_gender_fold0

* transfer_learning: store the experiments for training and cross-validating the efficentnet tranfer learning model.
  * effecientnet_implementation.ipynb
  * final_efficientnet_implementation.ipynb
  * plots.ipynb

## Results directory
All the results of all tasks are located in their respective folders after running the code.

## Required Libraries
- tensorflow==2.x
- numpy
- pickle5
- math
- sklearn
- tqdm
- pandas
- PIL

## Usage Guide
#### Data Preprocessing
1. Download the dataset from https://talhassner.github.io/home/projects/Adience/Adience-data.html

2. On your local machine, place the gender_preprocessing.ipynb and dataset in the following folder tree structure:
```
* root
  |
  |
  * gender_preprocessing.ipynb
  |
  |
  * Folds
    | 
    * original_txt_files : folder containing the txt labels from the 
    | above website
    | 
    * train_test_splitted: folder contain intermediate .csv from the     
    |  preprocessing preprocessing
    |
  |
  | 
  * Adience
    |
    * aligned : folder of images downloaded from the above dataset
  |
  |
  * serialized
    |
    * gender : folder containing the preprocessed pickle file serialized.
```
3. If the script fail to run, make sure you create the folders if they did not exist. 

Running this notebook should produce the train, test, train subset and test set for cross-validation and optimal model training

#### Google Colab Setup

1. Subscribe to Colab Pro (can use any random 5 number for US postal code).

2. Upload the pickle/serialized file and the notebook to colab.

3. In each of the notebook, adopt the link and path suitable to your folder structure. We have added abundant comment 
for you to figure this out.

##### Lean CNN Hyperparamters Tuning

The order of our experiments are baseline model, depth models, width models, dropout models and optimizer models

1. Cross-validation 5-fold of the various Hyperparamters models.
- The notebook order to run will be baseline_gender_fold0.ipynb, depth_experiments_fold0.ipynb, width_experiment_gender_fold0.ipynb, dropout_gender_fold0.ipynb
and optimizer_gender_fold0.ipynb
- Before running, make the change to fold number and bag number (the iteration and the data fold we are training or test on, across the five fold).
- Make the change to the Hyperparamters combination you want to do cross-validation
- Make the necessary changes to the serialzed dataset path if you have not done so.
- Run the notebook for model training. You may spawn multiple instances of the same type of notebook and change the fold, bag to your 
experiment order or preferences.
- To collect the best validation accuracy for that bag/ fold: Wait for model to finished training, then take the last epoch - 30 = epoch of minimum validation loss.
- Search from epoch = 1 to this epoch number for the max validation accuracy (should already be printed out from verbosity of checkpoint api of keras).

2. Train the optimal model.
- We have 3 notebooks optimal_baseline_gender.ipynb, optimal_width_expreiment_gender.ipynb and optimal_dropout_gender.ipynb, that should be run in order.
- For our experiments, the baseline model, width models and dropout models have optimal Hyperparamters change and therefore need to be trained and test on the final
train and test set. If there the results of depth experiments and optimizer experiments improve on your end for whatever reason, it is easy to craft the notebook to train
these optimal model.
- To adapt, take a look at how we adapt the baseline_gender_fold0 and optimal_baseline_gender or others and make the corresponding change.
- Make sure the path and links to dataset or where you save the results is as of your preferences
- Make sure the optimal hyperparameters are set in the variables of the notebooks.
- Train the models and wait for the results to be generated.

##### EfficientNet Transfer Learning
1. Cross-valdiation of Effecient Net transfer learning model:
- Similar approach to step 1 of Lean CNN as shown above. The notebook you should be looking at is inside transfer_learning, named as efficientnet_implementation.ipynb
2. Training of final model for transfer learning and obtain the graph
- Similar approach to step 2 of Lean CNN as shown above. The notebook you should be lookin at is final_efficientnet_implementation
