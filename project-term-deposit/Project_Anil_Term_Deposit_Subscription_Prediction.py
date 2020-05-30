#!/usr/bin/env python
# coding: utf-8

# ## Predicting Term deposit subscription
# 
# We need to build a model that will help the marketing team identify potential customers who are relatively more likely to subscribe term deposit and thus increase their hit ratio.

# In[118]:


# importing required libraries
import math

import pandas as pd # For data processing, CSV file I/O (e.g. pd.read_csv())
import numpy as np # For Linear Algebra
import pandas_profiling as pf # Generates profile reports from pandas DataFrame

#importing Machine Learning parameters and classifiers 
from sklearn import preprocessing # provides several common utility functions and transformer classes to change raw feature vectors into a representation more suitable for the downstream estimators
from sklearn.linear_model import LogisticRegression # Logistic Regression (aka logit, MaxEnt) classifier
from sklearn.tree import DecisionTreeClassifier # 
from sklearn.feature_selection import RFE # Feature ranking with recursive feature elimination.
from sklearn import metrics #  includes score functions, performance metrics and pairwise metrics and distance computations
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # Build a text report showing the main classification metrics,Compute confusion matrix to evaluate the accuracy of a classification. 
from sklearn.model_selection import train_test_split # splits data into random train and test subsets 
from sklearn.preprocessing import LabelEncoder


#Ensemble classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

#Visulization Libraries
from IPython.display import Markdown, display
import seaborn as sns # data visualization lib based upon matplotlib
import matplotlib.pyplot as plt # is a state-based interface to matplotlib. It provides a MATLAB-like way of plotting
get_ipython().run_line_magic('matplotlib', 'inline')
# a magic function which sets the backend of matplotlib to the 'inline' backend


# #### Import dataset

# In[2]:


dfBankData = pd.read_csv('bank-full.csv')


# ## Deliverable 1-  Exploratory data quality report 

# ### Univariate Analysis
# 

# ###### 1-a 
# Data types and description of the independent attributes which should include (name, meaning, range of values observed, central values (mean and median), standard deviation and quartiles, analysis of the body of distributions / tails, missing values, outliers

# In[3]:


dfBankData.head(20)


# In[4]:


dfBankData.shape


# In[5]:


# Data types present in data
dfBankData.dtypes


# In[6]:


dfBankData.describe().T


# In[7]:


dfBankData.info()


# In[8]:


print("Are there any null values ? : ", dfBankData.isnull().values.any())
print("Are there any na values ?   : ", dfBankData.isna().values.any())

print("\n")
print("------------Checking for null ------------------")
print(dfBankData.isnull().sum())
print("------------Cheking for NA ---------------------")
print(dfBankData.isna().sum())


# There no null or na values in dataset

# In[9]:


# Get unique values for all colums
for col in dfBankData.columns:
    print('Col Name {0}: Unique values {1}'.format(col, dfBankData[col].nunique()))


# In[10]:


# Finding outliers using Inter-Quartile Range which difference between 75th and 27th percentiles
# IQR = Q₃ − Q₁
# https://en.wikipedia.org/wiki/Interquartile_range
cols =['age','balance','day', 'duration', 'campaign', 'pdays', 'previous']
outliersCols=[]
for col in cols:
    q1 = dfBankData[col].quantile(0.25)
    q3 = dfBankData[col].quantile(0.75)
    iqr = q3-q1
    lower_range = q1-(1.5*iqr)
    upper_range = q3+(1.5*iqr) 
    isOutlier = (dfBankData.loc[(dfBankData[col] < lower_range)|(dfBankData[col] > upper_range)]).empty
    if isOutlier:
        display(Markdown("There are no outliers in {0}".format(col)))#("There are no outliers in {0}".format(col))
    else:
        print("There are outliers in {0}".format(col))
        outliersCols.append(col)
        
print(outliersCols)

noOfRows=(int)(len(outliersCols)/2)
noOfCols=2
fig, axs = plt.subplots(nrows = noOfRows, ncols = noOfCols, figsize=(15,15))
colIndex = 0         
for outlierCol in outliersCols: 
    sns.set()
    sns.boxplot(dfBankData[outlierCol], ax = axs[math.floor(colIndex/noOfCols)][colIndex % noOfCols])
    colIndex += 1 
        


# In[11]:


plt.figure()
dfBankData.hist(bins=20, figsize=(15,10), color='red')
plt.show()


# In[12]:


cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
noOfRows=(int)(len(cat_cols)/3)
noOfCols=3
fig, axs = plt.subplots(nrows = noOfRows, ncols = noOfCols, figsize=(15,15))
colIndex = 0    
plt.figure()
for catCol in cat_cols:
    sns.countplot(y=catCol, data=dfBankData, ax = axs[math.floor(colIndex/3)][colIndex % 3])
    colIndex += 1


# In[13]:


pf.ProfileReport(dfBankData)


# ## Observations
# - There are no null or na values in dataset
# - There are some values as 'unknown'
# - These are no outliers in feature day
# - There are outliers in following columns 
#     ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
# - Columns with unknown values :  [['job', 288], ['education', 1857], ['contact', 13020], ['poutcome', 36959]]

# ## 1b - Strategies to address the different data challenges such as data pollution, outlier’s treatment and missing values treatment.

# In[14]:


# Let's find categorical columns and numerical cols, we'll have to treat then differently
#Numerical cols
numericalCols = list(dfBankData.select_dtypes(exclude=['object']))
categoricalCols = list(dfBankData.select_dtypes(include=['object']))

print(" Numerical Columns are : ", numericalCols, "\n")
print(" Categorical Columns are : ", categoricalCols)


# - Most of outliers are in Numerical columns, so outlier treatment shall happen on ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
# - Most of unknown values are in categorical columns 

# In[15]:


for nu_col in numericalCols:
        print("******************************", nu_col, ":\n" )
        print(dfBankData[nu_col].value_counts(), "\n")


# In[16]:


# Let's find columns with unknown values and value counts 
colsWithUnknownVals = []
for col in dfBankData.columns:
    if 'unknown' in dfBankData[col].values:
        colsWithUnknownVals.append([col, dfBankData[dfBankData[col].str.contains('unknown')][col].count()])
        print("******************************", "\n")
        print("Values Count in - {0} - having unknown value : ".format(col))
        print(dfBankData[col].value_counts(), "\n")
print("************************************************", "\n")        
print("Columns with unknown values : \n", colsWithUnknownVals)
print("************************************************", "\n")         


# - Treatment of unknown values 
# - Job and education columns 'unknown' values should be replaces with appropriate values considering age, job and education
# - For exaple if job is 'housemaid' education columns 'unknown' can be replaced as 'primary'
# - If age >60 Job's 'unknown' value can be replaced as 'retired'
# - And if no other criteria found 'unknown' can be replaced with first of categorical values or most ocurring categorical value of that column
# 
# - Dealing with outliers
# - Put values in bucket of ranges, for example age 30,40,50,60 
# - Call Duration <10sec can be dropped as these are extreme values for this column
# 

# ## Multivariate analysis
# 
# a.Bi-variate analysis between the predictor variables and target column. Comment on your findings in terms of their relationship and degree of relation if any. 
# 
# Visualize the analysis using boxplots and pair plots, histograms or density curves. Select the most appropriate attributes.
# 
# b.Please provide comments in jupyter notebook regarding the steps you take and insights drawn from the plot

# In[17]:


# Tagert column - 'Target'
# Numerical columns - Numerical Columns are :  ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'] 
# Categorical Columns are :  ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']


# In[18]:


# sns pair plot
sns.pairplot(dfBankData)


# In[19]:


cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
noOfRows=(int)(len(cat_cols)/3)
noOfCols=3
fig, axs = plt.subplots(nrows = noOfRows, ncols = noOfCols, figsize=(25,25))
colIndex = 0    
plt.figure()
for catCol in cat_cols:
    sns.countplot(x=catCol, hue='Target', data=dfBankData, ax = axs[math.floor(colIndex/3)][colIndex % 3])
    colIndex += 1


# In[20]:


plt.figure(figsize = (10,10))
sns.heatmap(dfBankData.corr(), annot = True)


# In[21]:


plt.figure(figsize = (10,8))
sns.scatterplot(dfBankData['age'], dfBankData['balance'], hue = dfBankData['Target'])


# In[22]:


plt.figure(figsize = (18,16))
dfBankData.hist();


# In[23]:


plt.figure(figsize = (10,8))
sns.boxplot(x= dfBankData['age'], y=dfBankData['education'], hue=dfBankData['Target'])


# In[24]:


plt.figure(figsize = (10,8))
sns.boxplot(x= dfBankData['duration'], y=dfBankData['month'], hue=dfBankData['Target'])


# In[159]:


# TODO Comments 


# # 2 Prepare the data for analytics
# 
# 1.Ensure the attribute types are correct. If not, take appropriate actions.
# 
# 2.Get the data model ready. 
# 
# 3.Transform the data i.e. scale / normalize if required
# 
# 4.Create the training set and test set in ratioof 70:30
# 

# In[25]:


# There are mmany columns with type 'object'
categoricalCols = list(dfBankData.select_dtypes(include=['object']))
for catCol in categoricalCols:
    dfBankData[catCol] =dfBankData[catCol].astype('category')

dfBankData.dtypes


# In[26]:


dfBankData.groupby('Target').count()


# In[27]:


def bucketing_balance(data):
    data.loc[data['balance'] <= 72, 'balance'] = 1
    data.loc[(data['balance'] > 72) & (data['balance'] <= 1428), 'balance' ] = 2
    data.loc[(data['balance'] > 1428) & (data['balance'] <= 3462), 'balance' ] = 3
    data.loc[(data['balance'] > 3462) & (data['balance'] <= 102127), 'balance' ] = 4
    return data

bucketing_balance(dfBankData)


# In[28]:


def bucketing_education(df):
    df.loc[(df['age']>60) & (df['job']=='unknown'), 'job'] = 'retired'
    df.loc[(df['education']=='unknown') & (df['job']=='management'), 'education'] = 'tertiary'
    df.loc[(df['education']=='unknown') & (df['job']=='services'), 'education'] = 'secondary'
    df.loc[(df['education']=='unknown') & (df['job']=='housemaid'), 'education'] = 'primary'

    df.loc[(df['job'] == 'unknown') & (df['education']=='basic.4y'), 'job'] = 'blue-collar'
    df.loc[(df['job'] == 'unknown') & (df['education']=='basic.6y'), 'job'] = 'blue-collar'
    df.loc[(df['job'] == 'unknown') & (df['education']=='basic.9y'), 'job'] = 'blue-collar'
    df.loc[(df['job']=='unknown') & (df['education']=='professional.course'), 'job'] = 'technician'
    

bucketing_education(dfBankData)


# In[29]:


dfBankData['job'] = dfBankData.job.replace('unknown',dfBankData.job.mode()[0])
dfBankData['education'] = dfBankData.education.replace('unknown',dfBankData.education.mode()[0])


# In[30]:


#putting age into bins
def bucketing_age(df):
    df.loc[df["age"] < 30,  'age'] = 20
    df.loc[(df["age"] >= 30) & (df["age"] <= 39), 'age'] = 30
    df.loc[(df["age"] >= 40) & (df["age"] <= 49), 'age'] = 40
    df.loc[(df["age"] >= 50) & (df["age"] <= 59), 'age'] = 50
    df.loc[df["age"] >= 60, 'age'] = 60

bucketing_age(dfBankData)


# In[31]:


dfBankData.head(10)


# In[32]:


dfBankData['duration'] = dfBankData['duration'].apply(lambda n:n/60).round(2)


# In[33]:


print('Rows count having call duration less than 10 Sec -\t',dfBankData[dfBankData.duration < 10/60]['duration'].count())


# In[34]:


# drop rows where call duration was less than 10 seconds
#dropped 342 rows
dfBankData = dfBankData.drop(dfBankData[dfBankData.duration < 10/60].index, axis = 0, inplace = False)


# In[35]:


# Transform the data i.e. scale / normalize if required
# Now we have to get dummy variables
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
dfBankData = pd.get_dummies(data = dfBankData, columns = cat_cols, drop_first = True)

dfBankData.head()


# In[44]:


# We need to trasnform month column 


# In[87]:


label_encoder = LabelEncoder()
dfBankData['month'] = label_encoder.fit_transform(dfBankData['month'])
dfBankData['Target'] = label_encoder.fit_transform(dfBankData['Target'])


dfBankData.head(10)


# In[105]:


plt.figure(figsize = (20,18))
sns.heatmap(dfBankData.corr(), annot = True)


# In[94]:


# Create the training set and test set in ratioof 70:30


# In[95]:


X = dfBankData.drop('Target', axis = 1)
y = dfBankData['Target']
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3, random_state=0)


# In[96]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ### Todo Comments

# - TODO
# 
# - TODO
# 
# - TODO
# 

# # Deliverable –3  create the ensemble model
# 
# 1.First create model susing Logistic Regression and Decision Tree algorithm. Note the model performanceby using different matrices. Use confusion matrix to evaluate class level metrics i.e. Precision/Recall. Also reflect the accuracy and F1scoreof themodel.
# 
# 2.Build the ensemble models(Bagging and Boosting)and note the model performanceby using different matrices. Use same metrics as in above model. (at least 3 algorithms)
# 

# ### Model creating using Logistic Regression algorithm

# In[97]:


mLogReg = LogisticRegression()
mLogReg.fit(X_train, y_train)

y_pred_mLogReg = mLogReg.predict(X_test)

print("Logistirc Regression Model score on training data: {} \n".format(mLogReg.score(X_train, y_train)))
print("Logistirc Regression Model score on test data: {} \n".format(mLogReg.score(X_test, y_test)))
print("Confusion Matrics of Logistirc Regression Model :  \n \n",  confusion_matrix(y_test, y_pred_mLogReg))
roc_


# In[98]:


print(classification_report(y_test, y_pred_mLogReg))


# ### Model creating using Decision Tree algorithm

# In[99]:


modelDecisionTree_entropy = DecisionTreeClassifier(criterion='gini', random_state = 100,) 
modelDecisionTree_entropy.fit(X_train, y_train)

y_pred_dt_entropy = modelDecisionTree_entropy.predict(X_test)

print("Decision Tree Model score on training data: {} \n".format(modelDecisionTree_entropy.score(X_train, y_train)))
print("Decision Tree Model score on test data: {} \n".format(modelDecisionTree_entropy.score(X_test, y_test)))
print("Confusion Matrics of Decision Tree Model :  \n \n",  confusion_matrix(y_test, y_pred_dt_entropy))


# In[100]:


print(classification_report(y_test, y_pred_dt_entropy))


# In[101]:


# pruning the decision tree 

modelDT_entropy_pruned = DecisionTreeClassifier(criterion='gini', max_depth = 3, random_state = 100, min_samples_leaf = 5) 
modelDT_entropy_pruned.fit(X_train, y_train)

y_pred_dt_pruned = modelDT_entropy_pruned.predict(X_test)

print("Pruned Decision Tree Model score on training data: {} \n".format(modelDT_entropy_pruned.score(X_train, y_train)))
print("Pruned Decision Tree Model score on test data: {} \n".format(modelDT_entropy_pruned.score(X_test, y_test)))
print("\n")
print("Pruned Confusion Matrics of Decision Tree Model :  \n \n",  confusion_matrix(y_test, y_pred_dt_pruned))
print(classification_report(y_test, y_pred_dt_pruned))


# In[135]:


accuracies = {}


# # 13.  Bagging Classifier Algorithm 

# In[136]:


from sklearn.ensemble import BaggingClassifier

model_bgcl = BaggingClassifier(n_estimators = 200,max_samples= .7, bootstrap=True, oob_score=True, random_state = 22)
model_bgcl.fit(X_train, y_train)

model_bgcl_predict = model_bgcl.predict(X_test)
acc_model_bgcl = accuracy_score(y_test, model_bgcl_predict) * 100

accuracies['BaggingClassifer'] = acc_model_bgcl
accuracies['BaggingClassifer']


# In[137]:


from sklearn.ensemble import GradientBoostingClassifier
gbcl = GradientBoostingClassifier(n_estimators = 200,learning_rate = 0.1, random_state = 22)
gbcl = gbcl.fit(X_train, y_train)

gbcl_predict = gbcl.predict(X_test)
acc_gbcl = accuracy_score(y_test, gbcl_predict) * 100
accuracies['GradientBoostingClassifier'] = acc_gbcl
accuracies['GradientBoostingClassifier']


# In[138]:


from sklearn.ensemble import AdaBoostClassifier
abcl = AdaBoostClassifier(n_estimators = 200, learning_rate = 0.1, random_state = 22)
abcl = abcl.fit(X_train, y_train)

abcl_predict = abcl.predict(X_test)
acc_abcl = accuracy_score(y_test, abcl_predict) *100
accuracies['ADA'] = acc_abcl
accuracies['ADA']


# In[139]:


accuracies


# In[ ]:




