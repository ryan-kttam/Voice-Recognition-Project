# Machine Learning Classification
# Support Vector Machine
# outline:
# 1. data exploration
# 2. data standardization
# 3.

# Version 1

# 1. data exploration
import numpy as np , pandas as pd , matplotlib.pyplot as plt

# import the data
data = pd.read_csv("voice.csv")

# take a look at the data and its summary
data.head()
data.describe()

# check if there are any null values
data[data.isnull().any(axis=1)]
data.isnull().sum()

# counts for males and females
data['label'].value_counts() # there are 1584 male voices and 1584 female voices.

# visualize features relationship
import seaborn
seaborn.pairplot(data[['meanfreq', 'centroid', 'IQR', 'label']], hue = 'label', size = 2)

# get the column names, if interested
list(data)

# replace male as 1 and female as 0
data = data.replace( {'label': {'male': 1, 'female' :0}} )

# separating features and the label
label = data['label']
features = data.drop('label', axis = 1)
features.head()

# 2. data Standardization
#standardizing the data. (so that each attribute will have zero mean and sd of 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)
# alternative version
# scaler.fit(features)
# features = scaler.transform(features)

# split the data into training set and test set,
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split( features, label, test_size=0.1 , random_state = 1 )

# Support Vector Machine
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, accuracy_score

# finding the best C when kernel = 'rbf'
C_range = list(range(1,100))
c_tuning_score = []
k_fold = KFold(n_splits=10, random_state=1,shuffle=True) # object for 10-fold cv
scorer = make_scorer(accuracy_score) # scorer for cv, accuracy_score in this case
for c in C_range:
   model = SVC(C = c)
   cv_scores = cross_val_score(model, x_train, y_train, cv=k_fold, scoring=scorer)
   c_tuning_score.append(cv_scores.mean())
# plot the c_tuning_scores
import matplotlib.pyplot as plt
plt.plot(range(1,101),c_tuning_score)
plt.ylabel('cross-validated accuracy')
# we can see that the maximum is around C = 5-15,
plt.plot(range(1,16),c_tuning_score[1:16])
# in the above plot we can see that the maximum of C is 10.
# Let's break down 10 and see what is the exact value for C will be best for our prediction.
C_range2 = np.arange(9.0,11.0,0.1)
c_tuning_score2 = []
for c in C_range2:
   model = SVC(C = c)
   cv_scores = cross_val_score(model, x_train, y_train, cv=k_fold, scoring=scorer)
   c_tuning_score2.append(cv_scores.mean())
plt.plot(np.arange(9.0, 11.0, 0.1), c_tuning_score2)
# C = 10 is th best
# next is to find the best g.
g_range = [0.0001,0.001,.01,.1,1,10,100]
g_tuning_score = []
for g in g_range:
   model = SVC(gamma= g, C=10)
   cv_scores = cross_val_score(model, x_train, y_train, cv=k_fold, scoring=scorer)
   g_tuning_score.append(cv_scores.mean())
plt.plot(g_range, g_tuning_score)
# gamma significantly decreases its performance when it is > 10, so we zoom in and see which performed the best
plt.plot(g_range[1:5], g_tuning_score[1:5])
# As we can see from the plot, gamma peak at 0.1.
# our next step is to find the exact value of gamma, by zooming in even more.
g_range2 = np.arange(0.01,0.5,0.01)
g_tuning_score2 = []
for g in g_range2:
   model = SVC(gamma= g, C=10)
   cv_scores = cross_val_score(model, x_train, y_train, cv=k_fold, scoring=scorer)
   g_tuning_score2.append(cv_scores.mean())
plt.plot(g_range2, g_tuning_score2)
# we can see that gamma peak at around 0.2 - 0.3, we need to zoom to get the exact value
g_range3 = np.arange(0.02,0.1,0.001)
g_tuning_score3 = []
for g in g_range3:
   model = SVC(gamma= g, C=10)
   cv_scores = cross_val_score(model, x_train, y_train, cv=k_fold, scoring=scorer)
   g_tuning_score3.append(cv_scores.mean())
plt.plot(g_range3, g_tuning_score3)
# zoom in even more
plt.plot(g_range3[30:40], g_tuning_score3[30:40])
# the best gamma we have got is 0.055

# plug in the best parameter: C = 10 and gamma = 0.055,
clf = SVC(C=10, gamma = 0.055)
k_fold = KFold(n_splits=10, random_state=1,shuffle=True)
scorer = make_scorer(accuracy_score)
cv_scores = cross_val_score(clf, x_train, y_train, cv=k_fold, scoring=scorer)
cv_scores.mean() # the average accuracy is 98.04%

clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
accuracy_score(y_test,predictions) #accuracy: 99.05% on test set

# try a different split from the data, this time only 80% training data
x_train, x_test, y_train, y_test = train_test_split( features, label, test_size=0.2 , random_state = 2 )
clf = SVC(C=10, gamma = 0.055)
k_fold = KFold(n_splits=10, random_state=2, shuffle=True)
scorer = make_scorer(accuracy_score)
cv_scores = cross_val_score(clf, x_train, y_train, cv=k_fold, scoring=scorer)
cv_scores.mean() # the average accuracy is 98.11%
#predict the test set
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
accuracy_score(y_test, predictions) #accuracy: 97.95% on test set

# future improvement: try kernel = 'linear' or 'poly'
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train,y_train)
predictions = clf.predict(x_test)
accuracy_score(y_test, predictions)

# training the model with decision tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

# evaulating the DT model.
from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_test, predictions)
accuracy_score(y_test, predictions) # the accuracy is 98% 3 false positive and 3 false negative

# transformation guide:
# For Right-skewed (clustered at lower values),
# then try log, square root, etc, to move DOWN the ladder of power.
# For Left-skewed (clustered at high values),
# then try square, cube to move UP the ladder of power.



