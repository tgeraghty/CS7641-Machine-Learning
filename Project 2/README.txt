CS7641 Machine Learning
Fall 2019
Timothy Geraghty

Project 2

Link to Github Repo: https://github.com/tgeraghty/CS7641-Machine-Learning/tree/master/Project%202

Dataset: Online News Popularity
Source: https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity
Notebook: Project 2 - Online News Popularity

Links used for feature engineering for online news popularity dataset:
https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
https://people.duke.edu/~rnau/rsquared.htm

Links used for randomized optimization algorithms and toy problems:
https://buildmedia.readthedocs.org/media/pdf/mlrose/stable/mlrose.pdf
https://mlrose.readthedocs.io/en/stable/

Libaries used:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import optimizers
import time
import mlrose
import random

Running the Software:
To run the software, open the Jupyter notebook and select 'Kernel' and 'Restart & Run All' from the ribbon
