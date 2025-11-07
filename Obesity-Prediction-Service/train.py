#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score,accuracy_score,roc_auc_score,confusion_matrix,classification_report
from sklearn.feature_extraction import DictVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,export_text
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

#Read the dataset
df = pd.read_csv('../data/ObesityDataSet.csv')

#Basic data cleaning - 
# removing duplicates 
# filling missing values with 0 if any
print("Number of duplicated rows in the dataframe:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("Successfully removed duplicates. New shape of dataframe:", df.shape)
df = df.fillna(0)

# Standardizing column names and categorical values
df.columns = df.columns.str.replace(' ', '_').str.lower()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(' ', '_').str.lower()
df.isnull().sum()

#split the data into fulltrain and test sets
df_fulltrain,df_test=train_test_split(df, test_size=0.2, random_state=42)
df_fulltrain = df_fulltrain.reset_index(drop=True)
y_fulltrain = df_fulltrain['nobeyesdad']
del df_fulltrain['nobeyesdad']

#define directory
directory = 'model'

def train_model(df,y,C):
    pipeline = make_pipeline(
        DictVectorizer(),
        LogisticRegression(C=C,solver='newton-cg', penalty='l2', max_iter=1000 ,random_state=42)
    )
    dicts = df.to_dict(orient='records')
    pipeline.fit(dicts, y)  
    print("Model training completed.")
    return pipeline

def save_model(model, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, filename), 'wb') as f_out:
        pickle.dump(model, f_out)
        print(f"Model saved to {os.path.join(directory, filename)}")

pipeline = train_model(df_fulltrain, y_fulltrain, C=10)
save_model(pipeline, 'logistic_regression_model.bin' )

