#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score,accuracy_score,roc_auc_score,confusion_matrix,classification_report
from sklearn.feature_extraction import DictVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,export_text
import matplotlib.pyplot as plt
import seaborn as sns



# In[28]:


df = pd.read_csv('data/ObesityDataSet.csv')
df.isna().sum()


# There are no null values in the columns but there are 24 duplicated rows.Lets remove them from the dataset

# In[29]:


print("Number of duplicated rows:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("Number of duplicated rows after removing:", df.duplicated().sum())
df.shape


# First lets understand the features of this Obesity dataset to explore about them. There are 17 columns namely,
# 
# Attributes related to eating habits
# * CALC Consumption of Alcohol
# * FAVC Frequent consumption of high caloric food
# * FCVC Frequency of consumption of vegetables
# * NCP  Number of main meals
# * SMOKE smoker or non-smoker
# * CH2O  Consumption of water daily
# * CAEC  Consumption of food between meals
# 
# Attributes related to physical condition
# * Age, Gender, Height and weight columns are self explanatory
# * family_history_with_overweight - does he have family history of overweight
# * SCC  Calories consumption monitoring
# * FAF   Physical activity frequency
# * TUE   Time using technology devices
# * MTRANS Transportation used
# * NObeyesdad Target variable predicts if he is normal or obese
# 
# gender                            0.210643
# caec                              0.156140
# family_history_with_overweight    0.156009
# calc                              0.107467
# mtrans                            0.079317
# favc                              0.063315
# scc                               0.038011
# smoke                             0.008509
# 

# In[30]:


df.columns = df.columns.str.replace(' ', '_').str.lower()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(' ', '_').str.lower()
df.isnull().sum()
categorical_cols = list(df.dtypes[df.dtypes == 'object'].index)
numerical_cols = list(df.dtypes[df.dtypes != 'object'].index)
numerical_cols


# There are no missing values in this dataset. Basic datacleaning is done 

# In[31]:


for c in categorical_cols:
    print("feature : ",c,df[c].unique())
for c in numerical_cols:
    print("feature : ",c,df[c].describe())
    print()


# Let's check the correlation coefficient for numerical values in the training dataset after doing the test split

# In[32]:


df_fulltrain,df_test=train_test_split(df, test_size=0.2, random_state=42)
df_train,df_val =train_test_split(df_fulltrain, test_size=0.25, random_state=42)
df_fulltrain = df_fulltrain.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)  
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train['nobeyesdad']
y_val = df_val['nobeyesdad']
y_test = df_test['nobeyesdad']  
y_fulltrain = df_fulltrain['nobeyesdad']



# Let us plot the distribution of target variable to understand the obesity levels of people

# In[33]:


plt.figure(figsize=(4, 4))
sns.histplot(y=df.nobeyesdad, bins=50)
plt.title('Distribution of Obesity Levels')
del df_train['nobeyesdad']
del df_val['nobeyesdad']
del df_test['nobeyesdad'] 

class_names = ['insufficient_weight', 'normal_weight', 'obesity_type_i', 'obesity_type_ii','obesity_type_iii','overweight_level_i','overweight_level_ii']
min=0
max=0
for c in class_names:
    sum = (df.nobeyesdad == c).sum()
    print(c, ":", sum)
    if min ==0 or sum < min:
        min = sum
    if sum > max:
        max = sum
print("Min samples in a class :", min)
print("Max samples in a class :", max)  
class_imbalance_ratio = max / min
print("Class Imbalance Ratio :", class_imbalance_ratio)   


# class imbalance ratio is 1.31 <2 =>well balanced dataset

# In[34]:


# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_train[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Feature-to-Feature Correlation Matrix")
plt.show()



# From the heatmap , it is clear that all values of |r|<0.5 ==> all numerical features are distinct and are not redundant.So we can retain all the numerical features. Let's calculate Mutual Information Score for categorical values.

# In[35]:


if 'nobeyesdad' in categorical_cols:
    categorical_cols.remove('nobeyesdad')
print(categorical_cols)
print(numerical_cols)
def mutual_info_score_series(series):
    return mutual_info_score(series,y_train)

mi=df_train[categorical_cols].apply(mutual_info_score_series)
mi.sort_values(ascending=False)


# From Mutual Information score it is clear that smoke  has very low MI score .But removing that feature had no effect on the roc score.
# Let's create a confusion matrix  and classification report to undertand recall,precision and f1 scores
# 

# In[36]:


Model_ROC_AUC_scores = []
def train(df,y,C):
    dv =DictVectorizer(sparse=False)
    dicts = df.to_dict(orient='records')
    X = dv.fit_transform(dicts)
    model = LogisticRegression(C=C,solver='newton-cg', penalty='l2', max_iter=1000 ,random_state=42)
    model.fit(X, y)     
    return dv, model

def predict(df, dv, model):
    dicts = df.to_dict(orient='records')
    X = dv.transform(dicts) 
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    return y_pred, y_pred_proba

dv, model = train(df_train, y_train, C=10)
y_pred, y_pred_proba = predict(df_val, dv, model)
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", np.round(accuracy, 3))
roc_auc = roc_auc_score(y_val, y_pred_proba,multi_class='ovr')
print("ROC AUC:", np.round(roc_auc, 3))
Model_ROC_AUC_scores.append(('Logistic Regression', roc_auc))

cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['insufficient_weight', 'normal_weight', 'obesity_type_i', 'obesity_type_ii','obesity_type_iii','overweight_level_i','overweight_level_ii'], yticklabels=['insufficient_weight', 'normal_weight', 'obesity_type_i', 'obesity_type_ii','obesity_type_iii','overweight_level_i','overweight_level_ii'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Let's check the model with cross-validation

# In[37]:


n_splits = 5

for C in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 5, 10, 20, 30, 40, 50,100,200]:
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(df_fulltrain):
        df_train = df_fulltrain.iloc[train_idx]
        df_val = df_fulltrain.iloc[val_idx]

        y_train = df_train.nobeyesdad.values
        y_val = df_val.nobeyesdad.values
        del df_train['nobeyesdad']
        del df_val['nobeyesdad']

        dv, model = train(df_train, y_train, C=C)
        y_pred,y_pred_proba = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred_proba,multi_class='ovr')
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# From cross-validation using Kfold it is clear that at C=10 or20, the roc_auc is close to optimum 0.986.So lets choose C=10 as our
# final regularization value
# Lets train a final model combining both training and validation dataset and calculate roc_score against test dataset

# In[38]:


y_fulltrain = df_fulltrain.nobeyesdad
del df_fulltrain['nobeyesdad']
dv,model = train(df_fulltrain, y_fulltrain, C=10)
y_test, y_test_proba = predict(df_test, dv, model)

roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr')
print("LogisticRegression Model ROC AUC with test data: ", np.round(roc_auc, 3))



# After training the model with both validation and training data, it is testing with test data , we get optimum result with roc_score of 1.0 Moving on to decisiontree to check the performance

# In[39]:


dv= DictVectorizer(sparse=True)
train_dict = df_train.to_dict(orient = 'records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient = 'records')
X_val = dv.transform(val_dict)

auc_scores = []
depths = range(1, 21)
for max_depth in depths:
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred) 
    roc_auc = roc_auc_score(y_val, dt.predict_proba(X_val), multi_class='ovr')
    print("Decision Tree with depth " \
    f"{max_depth} Accuracy:", accuracy,"ROC AUC:", roc_auc)

    auc_scores.append(roc_auc)

# Plotting the AUC scores
plt.figure(figsize=(10, 6))
plt.plot(depths, auc_scores, marker='o', linestyle='-', color='b', label='AUC Score')
plt.xlabel('Tree Depth')
plt.ylabel('AUC Score')
plt.title('AUC Score vs. Tree Depth for Decision Tree Classifier')
plt.grid(True)
plt.xticks(depths)  # Ensure we show all depth values on the x-axis
plt.legend()
plt.show()



# Decision trees parameter tuning max_depth=5, 6, 7 and 8 have high ROC score.Lets find out the best values for min_sample_leaves

# In[40]:


scores = []
for max_depth in [5, 6, 7, 8]:
    for min_samples_leaf in [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100, 200]:
        dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred) 
        roc_auc = roc_auc_score(y_val, dt.predict_proba(X_val), multi_class='ovr')
        scores.append({
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'accuracy': accuracy,
            'roc_auc': roc_auc
        })
        print(f"max_depth: {max_depth}, min_samples_leaf: {min_samples_leaf}, Accuracy: {accuracy}, ROC AUC: {roc_auc}")
df_scores = pd.DataFrame(scores)
pivot_table = df_scores.pivot(index='min_samples_leaf', columns='max_depth', values='roc_auc')
sns.heatmap(pivot_table, annot=True, fmt=".3f") 


# From the pivot table it is clear that for max_depth =8 and min_sample_leaves =15, we get best ROC_AUC 0.980 
# The model is tested with testdata and  the scores ROC_AUC score is 0.978 which is almost same as validation ROC_AUC. 
# This confirms that DecisionTree model is not overfitting and consistent in prediction

# In[41]:


dt = DecisionTreeClassifier(max_depth=7,min_samples_leaf=15, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_val)
accuracy = accuracy_score(y_val, y_pred) 
roc_auc = roc_auc_score(y_val, dt.predict_proba(X_val), multi_class='ovr')
print("Decision Tree with validationdata Accuracy:", accuracy,"ROC AUC:", roc_auc)
print(export_text(dt, feature_names=list(dv.get_feature_names_out())))
Model_ROC_AUC_scores.append(('Decision Tree', roc_auc))


# Now let's train Ensemble Forest Classifier model

# In[42]:


scores = []
for n in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred) 
    roc_auc = roc_auc_score(y_val, rf.predict_proba(X_val), multi_class='ovr')
    scores.append({
        'n_estimators': n,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    })
    print(f"n_estimators: {n}, Accuracy: {accuracy}, ROC AUC: {roc_auc}")
df_scores = pd.DataFrame(scores)
plt.plot(df_scores['n_estimators'], df_scores['roc_auc'], marker='o')
plt.xlabel('Number of Estimators')  
plt.ylabel('ROC AUC')
plt.title('Random Forest Classifier ROC AUC vs Number of Estimators')
plt.show()  
print(export_text(rf.estimators_[0], feature_names=list(dv.get_feature_names_out())))


# Ensemble Random forest model gave best ROC_AUC of 0.991 with n_estimators =110.
# let's check with test data how the Random forest is performing

# In[43]:


rf = RandomForestClassifier(n_estimators=110, random_state=42)
rf.fit(X_train, y_train)

df_val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(df_val_dicts)
y_pred = rf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred) 
roc_auc = roc_auc_score(y_val, rf.predict_proba(X_val), multi_class='ovr')
print("Random Forest with Validation Data Accuracy:", np.round(accuracy, 3),"ROC AUC:", np.round(roc_auc, 3))
Model_ROC_AUC_scores.append(('Random Forest', np.round(roc_auc, 3)))

df_test_dicts = df_test.to_dict(orient='records')
X_test = dv.transform(df_test_dicts)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) 
roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test), multi_class='ovr')
print("Random Forest with TestData Accuracy:", np.round(accuracy, 3),"ROC AUC:", np.round(roc_auc, 3))



# Random Forest with Validation Data Accuracy: 0.892 ROC AUC: 0.991
# Random Forest with TestData Accuracy: 0.885 ROC AUC: 0.986
# This confirms that model is stable.
# Almost same ROC_AUC score can be seen for all the models that we have evaluated so far.
# Lets plot graph to evaluate the models

# In[44]:


Model, ROC_AUC_Score = zip(*Model_ROC_AUC_scores)
# Plotting the ROC AUC comparison using curves (line plot)
plt.figure(figsize=(10, 6))

# Plot the ROC AUC scores for each model
plt.plot(Model, ROC_AUC_Score, marker='o', linestyle='-', color='b', label='ROC AUC Score')

# Add labels and title
plt.xlabel('Model', fontsize=12)
plt.ylabel('ROC AUC Score', fontsize=12)
plt.title('Model ROC AUC Score Comparison - Curves', fontsize=14)

# Add a legend to identify the line
plt.legend(title='ROC AUC Scores', fontsize=10)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=10)

# Display grid for better readability
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()


# From the graph, it is clear that all models give the same ROC_AUC score.Let's finalize the simple LogisticRegression model
# for deployment

# In[47]:


for n in numerical_cols:
    print(df[n].describe())
    print()

for c in categorical_cols:
    print(df[c].value_counts())
    print()


# 
