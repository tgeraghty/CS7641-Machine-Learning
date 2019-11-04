CS7641 Machine Learning
Fall 2019
Timothy Geraghty

Project 3

Link to Github Repo: https://github.com/tgeraghty/CS7641-Machine-Learning/tree/master/Project%203

Dataset: Online News Popularity
Source: https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity
Notebook: Project 3 - Online News Popularity

Dataset: MNIST Handwritten Digits
Source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
Notebook: Project 3 - Digits

Libaries used:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import optimizers
import scipy
import time

# Set random seed 
import random
random.seed(27)

# Only warn once
import warnings
warnings.filterwarnings('ignore')

Running the Software:
To run the software, open the Jupyter notebook and select 'Kernel' and 'Restart & Run All' from the ribbon
