# Machine Learning Classification
# Support Vector Machine
# outline:
# 1. data exploration
# 2. data standardization and splitting data
# 3. Model training with default parameters
# 4. Model tuning
# 5. Test the final model with the test set

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

# 2. Data Standardization
# Data Standardization is a good practice to neutralize variables that have large values.
# Standardizing the data allows our training model to treat each features equally.
# standardizing the data using StandardScaler
# each attribute will have mean of 0 and sd of 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)
# alternative version
# scaler.fit(features)
# features = scaler.transform(features)

# split the data into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.1, random_state = 1 )

# 3. Model training using Support Vector Machine with default parameters
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import  accuracy_score , make_scorer
from sklearn.svm import SVC
clf = SVC()
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
accuracy_score(y_test, predictions)
# With default parameter, the accuracy on the test set is is 96.21%


# 4. Model tuning
# We will stick with kernel = 'rbf', and tune 'C' and 'gamma'
# starting with 'C'
# with C from 1 - 100
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
# The maximum for C is around between 5 to 15,
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
# C = 10 is the best

# Next we are going to find the best gamma.
# lets start from 0.0001 to 100
g_range = [0.0001, 0.001, .01, .1, 1, 10, 100]
g_tuning_score = []
for g in g_range:
   model = SVC(gamma= g, C=10)
   cv_scores = cross_val_score(model, x_train, y_train, cv=k_fold, scoring=scorer)
   g_tuning_score.append(cv_scores.mean())
plt.plot(g_range, g_tuning_score)
# Gamma significantly decreases its performance when it is > 10,
# so we zoom in and see which performed the best
plt.plot(g_range[0:5], g_tuning_score[0:5])
# As we can see from the plot, gamma peak at 0.1.
# our next step is to find the exact value of gamma, by zooming in even more.
g_range2 = np.arange(0.01, 0.2, 0.01)
g_tuning_score2 = []
for g in g_range2:
   model = SVC(gamma= g, C=10)
   cv_scores = cross_val_score(model, x_train, y_train, cv=k_fold, scoring=scorer)
   g_tuning_score2.append(cv_scores.mean())
plt.plot(g_range2, g_tuning_score2)
# the best gamma we have got is 0.05

# 5. Test the final model with the test set
# plug in the best parameter: C = 2 and gamma = 0.15
clf = SVC(C=10, gamma=0.05)

# Before we run on our test set, let's try a k-fold CV to see how it performs
k_fold = KFold(n_splits=10, random_state=3, shuffle=True)
scorer = make_scorer(accuracy_score)
cv_scores = cross_val_score(clf, x_train, y_train, cv=k_fold, scoring=scorer)
cv_scores.mean() # the k-fold cv average accuracy is 98.17%

# Now, run our final model with the test set!
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
accuracy_score(y_test,predictions) #accuracy: 99.05% on test set

# We got 99.05% accuracy on our test set. Not bad!
# There are some potential improvement about SVM, such as trying kernel = 'linear' or 'poly'
# but using 'rbf', we are able to generate generally decent accuracy percentage.






# other ML algorithm to be applied
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



