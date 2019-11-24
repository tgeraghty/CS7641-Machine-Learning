CS7641 Machine Learning
Fall 2019
Timothy Geraghty

Project 4

Link to Github Repo: https://github.com/tgeraghty/CS7641-Machine-Learning/tree/master/Project%204

Environment: OpenAI Gym Frozen Lake
Source: https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
Notebook: CS7641 Project 4 FrozenLake

Environment: OpenAI Gym Blackjack
Source: https://gym.openai.com/envs/Blackjack-v0/
Notebook: CS7641 Project 4 Blackjack

Libaries used:
# Import libraries to be used
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import time
import gym

# Set random seed 
import random
random.seed(27)

# Only warn once
import warnings
warnings.filterwarnings('ignore')

# Plot inline
%matplotlib inline

Running the Software:
To run the software, open the Jupyter notebook and select 'Kernel' and 'Restart & Run All' from the ribbon
