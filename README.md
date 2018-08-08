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

```
seaborn.pairplot(data[['meanfreq', 'centroid', 'IQR', 'label']], hue='label', size=2)
seaborn.pairplot(data[['kurt', 'Q25', 'Q75', 'label']], hue='label', size=2)
seaborn.pairplot(data[['minfun', 'maxfun', 'meandom', 'label']], hue='label', size=2)
```

## Data Preprocessing
Data Standardization is a good practice to neutralize variables that have large values. It can also minimize the risk of one feature from being too dominant compared to others. In other words, standardizing data allows machine learning models to treat each features equally. In this project, I will standardize the data by applying StandardScaler to each column: each attribute will have mean of 0 and sd of 1. 

in addition, I also use train_test_split from sklearn.model_selection to split the data into a training set and a test set, with 90% of the data as the training set, and the remaining 10% as the test set. 
```
scaler = StandardScaler()
features = scaler.fit_transform(features)
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.1, random_state = 1 )
```

## Model Training
After splitting the data into training and test set, I used support vector machine algorithm in order to predict Male/Female voices. I left the model with dafault parameters in order to see how well it will perform. 
In addition, I used 'accuracy_score' from sklearn.metrics as a metric to test the performance. It compares the predictions with the actual result: it will return the number of correct predictions over the number of predictions. The higher the accuracy (highest being as 1, lowest as 0), the better the performance. 
```
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
accuracy_score(y_test, predictions)
```
## Model tuning
It turns out that the SVM model with default parameters performed decently in the test set, with 96% accuracy. I, however, do not stop here. Model tuning is essential because assuming the default parameters are the best is not a good practice. In this case, I will adjust C and Gamma in order to find out the best parameters for this data; specifically, I will visualize and observe which 'C' will give us the best accuracy. 
```
C_range = list(range(1, 100))
c_tuning_score = []
k_fold = KFold(n_splits=10, random_state=1, shuffle=True) # object for 10-fold cv
scorer = make_scorer(accuracy_score) # scorer for cv, accuracy_score in this case
for c in C_range:
   model = SVC(C=c)
   cv_scores = cross_val_score(model, x_train, y_train, cv=k_fold, scoring=scorer)
   c_tuning_score.append(cv_scores.mean())
# visualize and observe which 'C' give us the best accuracy
import matplotlib.pyplot as plt
plt.plot(range(1, 100), c_tuning_score)
plt.ylabel('cross-validated accuracy')
plt.xlabel('parameter: C')
```
When using the code above, we can see that the model performs the best when C is in between 5 and 15. To find out the best estimate for C, I chose to reduce the range of the graph and to locate where the maximum of C is. 
```

plt.plot(range(5, 15), c_tuning_score[5:15])
# in the above plot we can see that the maximum of C is from 9 to 11.
# Let's break down the 10 and explore the best value for C.
C_range2 = list ( np.arange(9.00, 11, 0.01) )
c_tuning_score2 = []
for c in C_range2:
   model = SVC(C=c)
   cv_scores2 = cross_val_score(model, x_train, y_train, cv=k_fold, scoring=scorer)
   c_tuning_score2.append(cv_scores2.mean())
plt.plot(np.arange(9.00, 11.00, 0.01), c_tuning_score2)
```
It turns out that the model performs the best when C is 10. 
I then repeat similar technique with Gamma, with gamma equal (0.0001, 0.001, .01, .1, 1, 10, 100). It turns out the best Gamma is 0.15.
To summarize, the best hyperparameters for SVM for this data is C = 10 and gamma = 0.15.



