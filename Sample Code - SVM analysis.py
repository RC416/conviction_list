# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:51:22 2020

Four steps, not necessarily done in order. Each section should run on its own.

Part 1 - Build SVM model for training set (mostly illustrative for what used in parts 2-4)
Part 2 - Score drugs in the test set
Part 3 - Find optimal gamma value
Part 4 - Perform cross validation

Separate models are built for approved and unapproved drugs

@author: Ray
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import prep_and_analysis_functions as paf # custom analysis functions see file
from sklearn.model_selection import KFold

# load training set
training_set = pd.read_csv('training_set_standardized.csv')

'''
Part 1 - A - Prepare model for unapproved drugs
'''

#collect columns to input into model
feature_columns_unapproved = ['Fast_Track', 'SPA', 'Orphan', 'Breakthrough', 'RMAT', 'QIDP', 'large_public',
                   'small_public', 'phase_III_date_2015-2020', 'phase_III_date_2010-2014', 'phase_III_date_2000-2009',
                   'Infectious disease', 'Psychiatry', 'Oncology', 'Autoimmune/immunology','Dermatology',
                   'Neurology', 'Respiratory', 'Endocrine', 'Hematology', 'Cardiovascular', 'Not Specified',
                   'Gastroenterology (non inflammatory bowel disease)', 'Obstetrics/Gynecology', 'Metabolic',
                   'Allergy', 'Ophthalmology', 'Urology', 'ENT/Dental', 'Renal', 'Orthopedics',
                   'Rheumatology (non autoimmune)', 'likelihood %', 'diff_from_avg']

# create features and labels list
features_unapproved = training_set.loc[training_set['approved']==0, feature_columns_unapproved]
labels_unapproved = training_set.loc[training_set['approved']==0, 'Drug List Status']

# build training/testing set
x_train, x_test, y_train, y_test = train_test_split(features_unapproved, labels_unapproved, test_size=0.2)

# create classifier object
SVC_unapproved = SVC(kernel='rbf', gamma=1, probability=True) # see below for finding optimal gamma

# train model
SVC_unapproved.fit(x_train, y_train)
y_pred = SVC_unapproved.predict(x_test) # create predictions for test set

# quick accuracy check
paf.print_accuracy(y_test, y_pred)
paf.plot_confusion_matrix(y_test, y_pred)
paf.plot_confusion_matrix_normalized(y_test, y_pred)


'''
Part 1 - B - Prepare model for approved drugs
'''

#collect columns to input into model
feature_columns_approved = ['Fast_Track', 'SPA', 'Orphan', 'Breakthrough', 'RMAT', 'QIDP', 'large_public',
                   'small_public', 'Infectious disease', 'Psychiatry', 'Oncology', 'Autoimmune/immunology','Dermatology',
                   'Neurology', 'Respiratory', 'Endocrine', 'Hematology', 'Cardiovascular', 'Not Specified',
                   'Gastroenterology (non inflammatory bowel disease)', 'Obstetrics/Gynecology', 'Metabolic',
                   'Allergy', 'Ophthalmology', 'Urology', 'ENT/Dental', 'Renal', 'Orthopedics',
                   'Rheumatology (non autoimmune)', 'Actual_US_Approval_Date']

# create features and labels list
features_approved = training_set.loc[training_set['approved']==1, feature_columns_approved]
labels_approved = training_set.loc[training_set['approved']==1, 'Drug List Status']

# build training/testing set
x_train, x_test, y_train, y_test = train_test_split(features_approved, labels_approved, test_size=0.2)

# create classifier object
SVC_approved = SVC(kernel='rbf', gamma=1, probability=True) # see below for finding optimal gamma

# train model
SVC_approved.fit(x_train, y_train)
y_pred = SVC_approved.predict(x_test) # create predictions for test set

# quick accuracy check
paf.print_accuracy(y_test, y_pred)
paf.plot_confusion_matrix(y_test, y_pred)
paf.plot_confusion_matrix_normalized(y_test, y_pred)


'''
Part 2 - A - Score unapproved drugs from BMT

model is built without a holdout sample
'''

# load testing set and training set
test_data_unapproved = pd.read_csv('training_set_unapproved_standardized.csv')
training_set = pd.read_csv('training_set_standardized.csv')

#collect columns to input into model
feature_columns_unapproved = ['Fast_Track', 'SPA', 'Orphan', 'Breakthrough', 'RMAT', 'QIDP', 'large_public',
                   'small_public', 'phase_III_date_2015-2020', 'phase_III_date_2010-2014', 'phase_III_date_2000-2009',
                   'Infectious disease', 'Psychiatry', 'Oncology', 'Autoimmune/immunology','Dermatology',
                   'Neurology', 'Respiratory', 'Endocrine', 'Hematology', 'Cardiovascular', 'Not Specified',
                   'Gastroenterology (non inflammatory bowel disease)', 'Obstetrics/Gynecology', 'Metabolic',
                   'Allergy', 'Ophthalmology', 'Urology', 'ENT/Dental', 'Renal', 'Orthopedics',
                   'Rheumatology (non autoimmune)', 'likelihood %', 'diff_from_avg']

# create features and labels list
features_unapproved = training_set.loc[training_set['approved']==0, feature_columns_unapproved]
labels_unapproved = training_set.loc[training_set['approved']==0, 'Drug List Status']


# create classifier object
SVC_unapproved = SVC(kernel='rbf', gamma=1, probability=True)

# train model
SVC_unapproved.fit(features_unapproved, labels_unapproved)

# drop a weird blank row
test_data_unapproved = test_data_unapproved.dropna(subset=feature_columns_unapproved)

# use model to predict score for each drug
test_data_unapproved['SVM score'] = SVC_unapproved.predict_proba(test_data_unapproved.loc[:,feature_columns_unapproved])[:,1]

# use model to score drugs used in building the model
training_set_scored = training_set[training_set['approved']==0]
training_set_scored['SVM score'] = SVC_unapproved.predict_proba(features_unapproved)[:,1]

# save results to csv
test_data_unapproved.loc[:, ('DrugID', 'SVM score')].to_csv('test_set_unapproved_SVM_scored.csv', index=False)
training_set_scored.loc[:,('DrugID', 'SVM score')].to_csv('training_set_unapproved_SVM_scored.csv', index=False)

# plt.hist(test_data_unapproved['SVM score'],bins=25)


'''
Part 2 - B - Score approved drugs from BMT

model is built without a holdout sample
'''

# load testing set and training set
test_data_approved = pd.read_csv('training_set_approved_standardized.csv')
training_set = pd.read_csv('training_set_standardized.csv')


#collect columns to input into model
feature_columns_approved = ['Fast_Track', 'SPA', 'Orphan', 'Breakthrough', 'RMAT', 'QIDP', 'large_public',
                   'small_public', 'Infectious disease', 'Psychiatry', 'Oncology', 'Autoimmune/immunology','Dermatology',
                   'Neurology', 'Respiratory', 'Endocrine', 'Hematology', 'Cardiovascular', 'Not Specified',
                   'Gastroenterology (non inflammatory bowel disease)', 'Obstetrics/Gynecology', 'Metabolic',
                   'Allergy', 'Ophthalmology', 'Urology', 'ENT/Dental', 'Renal', 'Orthopedics',
                   'Rheumatology (non autoimmune)', 'Actual_US_Approval_Date']

# create features and labels list
features_approved = training_set.loc[training_set['approved']==1, feature_columns_approved]
labels_approved = training_set.loc[training_set['approved']==1, 'Drug List Status']

# create classifier object
SVC_approved = SVC(kernel='rbf', gamma=1, probability=True)

# train model
SVC_approved.fit(features_approved, labels_approved)

# drop any blank rows
test_data_approved = test_data_approved.dropna(subset=feature_columns_approved)

# use model to predict score for each drug
test_data_approved['SVM score'] = SVC_approved.predict_proba(test_data_approved.loc[:,feature_columns_approved])[:,1] # returns an array of pr for each outcome

# use model to score drugs used in building the model
training_set_scored = training_set[training_set['approved']==1]
training_set_scored['SVM score'] = SVC_approved.predict_proba(features_approved)[:,1]

# save results to csv
test_data_approved.loc[:, ('DrugID', 'SVM score')].to_csv('test_set_approved_SVM_scored.csv', index=False)
training_set_scored.loc[:,('DrugID', 'SVM score')].to_csv('training_set_approved_SVM_scored.csv', index=False)

# plt.hist(test_data_approved['SVM score'], bins=50)


'''
Part 3

Code to find optimal gamma value
for unapproved drugs: insensitive to values 0-1000, omit gamma specification
for approved drugs: 0.025 - 2.000, use value of 1
'''
# for unapproved drugs
#collect columns to input into model
feature_columns_unapproved = ['Fast_Track', 'SPA', 'Orphan', 'Breakthrough', 'RMAT', 'QIDP', 'large_public',
                   'small_public', 'phase_III_date_2015-2020', 'phase_III_date_2010-2014', 'phase_III_date_2000-2009',
                   'Infectious disease', 'Psychiatry', 'Oncology', 'Autoimmune/immunology','Dermatology',
                   'Neurology', 'Respiratory', 'Endocrine', 'Hematology', 'Cardiovascular', 'Not Specified',
                   'Gastroenterology (non inflammatory bowel disease)', 'Obstetrics/Gynecology', 'Metabolic',
                   'Allergy', 'Ophthalmology', 'Urology', 'ENT/Dental', 'Renal', 'Orthopedics',
                   'Rheumatology (non autoimmune)', 'likelihood %', 'diff_from_avg']

# create features and labels list
features_unapproved = training_set.loc[training_set['approved']==0, feature_columns_unapproved]
labels_unapproved = training_set.loc[training_set['approved']==0, 'Drug List Status']


gammas = [x/10 for x in range(1,200,5)]
scores = {}

for n in range(1,11):
    
    rep_scores = []
    
    for gamma in gammas:

        # build training/testing set
        x_train, x_test, y_train, y_test = train_test_split(features_unapproved, labels_unapproved, test_size=0.2)
        
        # create classifier object
        SVC_unapproved = SVC(kernel='rbf', probability=True) # see below for finding optimal gamma
        
        # train model
        SVC_unapproved.fit(x_train, y_train)
        y_pred = SVC_unapproved.predict(x_test) # create predictions for test set

        score=SVC_unapproved.score(x_test,y_test)
        rep_scores.append(score)
    
    scores[n] = rep_scores    

gammas_unapproved = pd.DataFrame.from_dict(scores, columns=gammas, orient='index')

    
plt.plot(gammas, gammas_unapproved.mean(axis=0))
plt.ylim(0.5,0.8)
plt.xlabel('gamma')
plt.ylabel('score')




# for approved drugs
#collect columns to input into model
feature_columns_approved = ['Fast_Track', 'SPA', 'Orphan', 'Breakthrough', 'RMAT', 'QIDP', 'large_public',
                   'small_public', 'Infectious disease', 'Psychiatry', 'Oncology', 'Autoimmune/immunology','Dermatology',
                   'Neurology', 'Respiratory', 'Endocrine', 'Hematology', 'Cardiovascular', 'Not Specified',
                   'Gastroenterology (non inflammatory bowel disease)', 'Obstetrics/Gynecology', 'Metabolic',
                   'Allergy', 'Ophthalmology', 'Urology', 'ENT/Dental', 'Renal', 'Orthopedics',
                   'Rheumatology (non autoimmune)', 'Actual_US_Approval_Date']

# create features and labels list
features_approved = training_set.loc[training_set['approved']==1, feature_columns_approved]
labels_approved = training_set.loc[training_set['approved']==1, 'Drug List Status']


gammas = [x/1000 for x in range(1,200,2)]
scores = {}

for n in range(1,11):
    
    rep_scores = []
    
    for gamma in gammas:

        # build training/testing set
        x_train, x_test, y_train, y_test = train_test_split(features_approved, labels_approved, test_size=0.2)
        
        # create classifier object
        SVC_approved = SVC(kernel='rbf', gamma=gamma, probability=True) # see below for finding optimal gamma
        
        # train model
        SVC_approved.fit(x_train, y_train)
        y_pred = SVC_approved.predict(x_test) # create predictions for test set

        score=SVC_approved.score(x_test,y_test)
        rep_scores.append(score)
    
    scores[n] = rep_scores    

gammas_approved = pd.DataFrame.from_dict(scores, columns=gammas, orient='index')

    
plt.plot(gammas, gammas_approved.mean(axis=0))
plt.xlabel('gamma')
plt.ylabel('score')




'''
Part 4 - test accuracy using k-fold cross validation
see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold
see: https://machinelearningmastery.com/k-fold-cross-validation/
see: https://www.codecademy.com/paths/data-science/tracks/dspath-supervised/modules/dspath-classification/articles/training-set-vs-validation-set-vs-test-set
'''

'''
Part 4 - A - unapproved drugs
'''

# load training set
training_set = pd.read_csv('training_set_standardized.csv')

#collect columns to input into model
feature_columns_unapproved = ['Fast_Track', 'SPA', 'Orphan', 'Breakthrough', 'RMAT', 'QIDP', 'large_public',
                   'small_public', 'phase_III_date_2015-2020', 'phase_III_date_2010-2014', 'phase_III_date_2000-2009',
                   'Infectious disease', 'Psychiatry', 'Oncology', 'Autoimmune/immunology','Dermatology',
                   'Neurology', 'Respiratory', 'Endocrine', 'Hematology', 'Cardiovascular', 'Not Specified',
                   'Gastroenterology (non inflammatory bowel disease)', 'Obstetrics/Gynecology', 'Metabolic',
                   'Allergy', 'Ophthalmology', 'Urology', 'ENT/Dental', 'Renal', 'Orthopedics',
                   'Rheumatology (non autoimmune)', 'likelihood %', 'diff_from_avg']

# create features and labels list
features = training_set.loc[training_set['approved']==0, feature_columns_unapproved].reset_index(drop=True)
labels = training_set.loc[training_set['approved']==0, 'Drug List Status'].reset_index(drop=True)

# create object to split dataset into n_splits
kfold = KFold(n_splits=5, shuffle=False) 

# create lists to store results
accuracy = []
recall = []
precision = []
f1_score = []


# loop through all splits and analyse model accuracy
for train_index, test_index in kfold.split(features):
    
    # create train/test splits
    x_train, x_test = features.loc[train_index,:], features.loc[test_index,:]
    y_train, y_test = labels[train_index], labels[test_index]
    
    
    # initiate model and train
    classifier = SVC(kernel='rbf', gamma=1, probability=True) # choose optimal k for dataset
    classifier.fit(x_train, y_train)
    
    # calculate predicted labels
    y_pred = classifier.predict(x_test)
    
    # check and record accuracy metrics
    accuracy.append(metrics.accuracy_score(y_test, y_pred))
    recall.append(metrics.recall_score(y_test, y_pred))
    precision.append(metrics.precision_score(y_test, y_pred))
    f1_score.append(metrics.f1_score(y_test, y_pred))
    
cv_results = pd.DataFrame(data={'accuracy':accuracy, 'recall':recall,
                                'precision':precision, 'f1 score':f1_score})

print(cv_results.mean(axis=0))
print(cv_results.std(axis=0))



'''
Part 4 - B - approved drugs
'''

# load training set
training_set = pd.read_csv('training_set_standardized.csv')

#collect columns to input into model
feature_columns_approved = ['Fast_Track', 'SPA', 'Orphan', 'Breakthrough', 'RMAT', 'QIDP', 'large_public',
                   'small_public', 'Infectious disease', 'Psychiatry', 'Oncology', 'Autoimmune/immunology','Dermatology',
                   'Neurology', 'Respiratory', 'Endocrine', 'Hematology', 'Cardiovascular', 'Not Specified',
                   'Gastroenterology (non inflammatory bowel disease)', 'Obstetrics/Gynecology', 'Metabolic',
                   'Allergy', 'Ophthalmology', 'Urology', 'ENT/Dental', 'Renal', 'Orthopedics',
                   'Rheumatology (non autoimmune)', 'Actual_US_Approval_Date']

# create features and labels list
features = training_set.loc[training_set['approved']==1, feature_columns_approved].reset_index(drop=True) # need to reset index for fix kfold, losing indeces is OK here where only accuracy is being tested
labels = training_set.loc[training_set['approved']==1, 'Drug List Status'].reset_index(drop=True) 

# create object to split dataset into n_splits
kfold = KFold(n_splits=10, shuffle=False) 

# create lists to store results
accuracy = []
recall = []
precision = []
f1_score = []


# loop through all splits and analyse model accuracy
for train_index, test_index in kfold.split(features):
    
    # create train/test splits
    x_train, x_test = features.iloc[train_index,:], features.iloc[test_index,:]
    y_train, y_test = labels[train_index], labels[test_index]
    
    # initiate model and train
    classifier = SVC(kernel='rbf', gamma=1, probability=True) # choose optimal k for dataset
    classifier.fit(x_train, y_train)
    
    # calculate predicted labels
    y_pred = classifier.predict(x_test)
    
    # check and record accuracy metrics
    accuracy.append(metrics.accuracy_score(y_test, y_pred))
    recall.append(metrics.recall_score(y_test, y_pred))
    precision.append(metrics.precision_score(y_test, y_pred))
    f1_score.append(metrics.f1_score(y_test, y_pred))
    
cv_results = pd.DataFrame(data={'accuracy':accuracy, 'recall':recall,
                                'precision':precision, 'f1 score':f1_score})

print(cv_results.mean(axis=0))
print(cv_results.std(axis=0))