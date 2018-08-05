# Voice-Recognition-Project
This project will use Machine Learning algorithms to determine a voice as a male or a female. The data has been preprocessed and transformed into various formats of frequency, such as mean frequency, centroid, etc. 

## Getting Started
This project requires Python 2 and needs the following packages:
```
import numpy as np, pandas as pd
import seaborn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import  accuracy_score , make_scorer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import confusion_matrix,accuracy_score
```
## Project Outline:
This project will have five sub-sections:
- 1. data exploration
- 2. data standardization and splitting data
- 3. Model training with default parameters
- 4. Model tuning
- 5. Test the final model with the test set

## Data Exploration
This dataset contains a total of 3168 rows and 21 columns. Males and Females are equally distributed in the dataset (1584 male recordings and 1584 female recordings. This dataset is also available at https://www.kaggle.com/jeganathan/voice-recognition. 

To visualize how different genders have influences on voice, I generated graphs that separate male and female. The graphs indicated that male tends to have lower values in most features. This is a good sign for machine learning algorithms, because two genders seem to have very different distributions in most features. 
