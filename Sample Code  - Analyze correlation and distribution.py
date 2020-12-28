# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:51:13 2020

input: training set scored with all models and testing set scored with all models

output: correlaton plots and distribution plots

@author: Ray
"""

import pandas as pd
import seaborn as sea
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import prep_and_analysis_functions as paf


training_set = pd.read_csv('training_set_scored_all_models.csv')
training_set_approved = training_set.loc[training_set['approved']==1,]
training_set_unapproved = training_set.loc[training_set['approved']==0,]

testing_set_approved = pd.read_csv('test_set_approved_scored_all_models.csv')
testing_set_unapproved = pd.read_csv('test_set_unapproved_scored_all_models.csv')

# create gridspace
grid = sea.PairGrid(data=testing_set_unapproved, vars=['KNN_score_r2', 'LR_score_r2', 'SVM_score_r2'],
                    height=2)

# fill in diagonal
grid = grid.map_diag(plt.hist, bins=10)

# fill in lower triangle
grid = grid.map_lower(plt.scatter, alpha=0.2)

# fill in correlation coefficient
def corr(x, y, **kwargs):  
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$=' + str(round(coef, 2))  
    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy = (0.08, 0.93), size = 12, xycoords = ax.transAxes)
    
grid = grid.map_lower(corr)

def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)
    
grid = grid.map_upper(hide_current_axis)





















