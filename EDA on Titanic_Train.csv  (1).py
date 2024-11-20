#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# # Our Data

# In[ ]:


#Reading titanic_train.csv into pandas dataframe


# In[4]:


train = pd.read_csv(r"C:\Users\Adrian\Downloads\train.csv")
train


# In[5]:


train.head()


# # Exploratory Data Analysis

# In[ ]:


#To get infor about our dataset


# In[6]:


train.info()


# In[ ]:


#Exploring all Unique values in our Dataset


# In[7]:


train.nunique()


# # Missing Data

# In[ ]:


# Using Seaborn to create a simple hitmap to discover where data is missing


# In[8]:


train.isnull()


# In[ ]:


#Exploring total null values in the dataset


# In[9]:


train.isnull().sum()


# In[10]:


sns.heatmap(train.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')


# In[ ]:


#There are missing data in columns Age, Cabin and Embarked.
#Too much data missing from the Cabin column as compared to Age Column.
#To later drop the missing Age data row and perform data cleaning to fill them.
#Let proceed with More Visualizations down here:


# In[11]:


sns.set_style('whitegrid')
sns.countplot(x = 'Survived', data= train)


# In[12]:


sns.set_style('whitegrid')
sns.countplot(x = 'Survived', hue = 'Sex', data= train, palette = 'RdBu_r')


# In[24]:


sns.set_style('whitegrid')
sns.countplot(x ='Survived', hue='Pclass', data=train, palette ='rainbow')


# In[13]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=40)


# In[14]:


train['Age'].hist(bins=30, color='darkred', alpha=0.3)


# In[15]:


sns.countplot(x='SibSp',data=train)


# In[16]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# # Data Cleaning

# In[ ]:


#Filling in missing age data instead of just dropping the missing data rows.
#One way to do this is by filling in the mean age of all the passengers(imputation).
#However we can be smarter about this and check the Passenger Class, For Example:


# In[17]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[ ]:


def impute_age(cols):
    
Age=cols[0]

Pclass=cols[1]

if pd.isnull(Age):

if Pclass == 1:
    return 37

elif Pclass == 2:
    return 29

else:
     return 24
else:
     return Age
    


# In[ ]:


#Applying the function


# In[25]:


train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)


# In[ ]:


#Checking the heatmap again


# In[28]:


sns.heatmap(train.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')


# In[ ]:


#Age null values should have been replaced if code run successfully
#Dropping the Cabin Column and the row in Embarked that is NaN


# In[26]:


train.drop('Cabin',axis=1,inplace=True)


# In[27]:


train.head()


# In[ ]:


train.dropna(inplace=True)


# # Converting Categorical Features

# In[ ]:


# There is need to convert categorical features to dummy variables using pandas! Otherwise our ML Algorithm won't be able to directly take in those
  features as inputs.


# In[30]:


train.info()


# In[32]:


pd.get_dummies(train['Embarked'],drop_first=True).head()


# In[33]:


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark =  pd.get_dummies(train['Embarked'],drop_first=True)


# In[34]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[35]:


train.head()


# In[36]:


train = pd.concat([train,sex,embark],axis=1)


# In[37]:


train.head()


# In[ ]:


# Data now ready for our model


# # Building a Logistic Regression Model

# In[ ]:


#Commenced by splitting Data into a training set and test set(...)


# # Train Test Split

# In[38]:


train.drop('Survived',axis=1).head()


# In[39]:


train['Survived'].head()


# In[40]:


from sklearn.model_selection import train_test_split


# In[42]:


x_train,x_test,y_train,y_test = train_test_split(train.drop('Survived',axis=1),
                                                 train['Survived'],test_size=0.30,
                                                 random_state=101)
                                          


# # Training and Predicting

# In[43]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)


# In[45]:


LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                  intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, 
                  penalty=12, random_state=None, solver='liblinear', tol = 0.0001,
                  verbose=0, warm_start=False)


# In[50]:


predictions = logmodel.predict(x_test)


# In[51]:


from sklearn.metrics import confusion_matrix


# In[52]:


accuracy=confusion_matrix(y_test,predictions)


# In[53]:


accuracy


# In[ ]:





# In[ ]:




