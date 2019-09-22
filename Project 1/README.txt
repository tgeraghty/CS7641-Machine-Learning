CS7641 Machine Learning
Fall 2019
Timothy Geraghty

Project 1

Link to Github Repo: https://github.com/tgeraghty/CS7641-Machine-Learning/tree/master/Project%201

Dataset: Online News Popularity
Source: https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity
Notebook: Project 1 - Online News Popularity

Dataset: Credit Card Fraud 
Source: https://www.kaggle.com/mlg-ulb/creditcardfraud
Notebook: Project 1 - Credit Card Fraud

Links used for feature engineering:
https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
https://people.duke.edu/~rnau/rsquared.htm

Links used for algorithm development: 
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
https://keras.io/getting-started/sequential-model-guide/

Links for plotting:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

Libraries Used:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm, linear_model
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import optimizers
import time

Running the Software:
To run the software, open the Jupyter notebook and select 'Kernel' and 'Restart & Run All' from the ribbon
