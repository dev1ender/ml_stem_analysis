
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[44]:


df = pd.read_csv("/home/dev/Desktop/Redcarpet/ml_stem_analysis-master/Federal.csv")


# In[45]:


df.shape


# In[46]:


df.head()


# In[47]:


#Deleting all the columns having missing value greater than 50
columns =df.columns[df.isnull().sum()<50]
df=df[columns]
df = df.drop(['A) Brief Description','Index Number', 'Investment Name','I1) STEM Learners Targeted? Specify.'], axis = 1)


# In[48]:


df.columns


# In[49]:


df.isnull().sum()


# ## stage 1

# In[50]:



#Rename the columns and preprocessing the data 
df = df.rename(columns={'D) Mission-specific or General STEM?':'MGSTEM','B) Year Established':'YE', ' C1) Funding FY2008 ': 'FY2008', ' C2) Funding FY2009 ': 'FY2009', ' C3) Funding FY2010 ': 'FY2010','I1) STEM Learners Targeted? Specify.': 'STEM'})
df['FY2008'] = pd.DataFrame(df['FY2008'].str.replace(",",""))
df['FY2009'] = pd.DataFrame(df['FY2009'].str.replace(",",""))
df['FY2010'] = pd.DataFrame(df['FY2010'].str.replace(",",""))
df['FY2008'] = pd.DataFrame(df['FY2008'].str.replace("-","0"))
df['FY2009'] = pd.DataFrame(df['FY2009'].str.replace("-","0"))
df['FY2010'] = pd.DataFrame(df['FY2010'].str.replace("-","0"))

#setting the type to float
df['FY2008']=df['FY2008'].astype(float)
df['FY2009']=df['FY2009'].astype(float)
df['FY2010']=df['FY2010'].astype(float)

#deleting the last row as it only contain the total of the fundings every year
df.drop(df.tail(1).index,inplace=True)

#filling the null values.
df['FY2008']=df['FY2008'].fillna(df['FY2008'].median())
df['FY2009']=df['FY2009'].fillna(df['FY2009'].median())
df['FY2010']=df['FY2010'].fillna(df['FY2010'].median())
df['Q) Legislation Required to Shift Focus?'] = df['Q) Legislation Required to Shift Focus?'].fillna(df['Q) Legislation Required to Shift Focus?'].value_counts().max())
df['Agency'] = df['Agency'].fillna(df["Agency"].value_counts().max())
#droping left over null values
df=df.dropna()


# ### STAGE 1:
# #1) Calculate % growth of funding between year 2008 & 2009.
# #2) If funding is positive, tag it as 1, if funding is negative tag it as 0. This is the target variable.

# In[51]:


#stage one both part performed
def add_target():
    Growth = ((df['FY2009'] - df['FY2008']) / df['FY2008']) * 100
    df['target'] = (Growth >= 0).astype(int)
    return df


# In[52]:


df = add_target()


# In[53]:


df.head()


# ## STAGE 2:
# #1) Create graphs of univariate distribution of all non funding variables and share on a jupyter notebook. Just FYI_Funding FY2008, FY2009, FY2010 are the "funding variables"
# #2) Calculate mutual_info_score of target variable created in stage 1 & ALL non funding variables and share on a jupyter notebook.

# In[54]:


sns.set(style="darkgrid")

fig = plt.figure(figsize=(10,6))
sns.countplot(x="Agency", data = df)
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Agency', fontsize=14)
plt.xticks(rotation='vertical')
plt.title("Frequency of Agency types", fontsize=15)
plt.show()


# In[55]:


fig = plt.figure(figsize=(18,6))
sns.countplot(x="Subagency", data = df)
plt.ylabel('Frequecy', fontsize=14)
plt.xlabel('Subagency', fontsize=14)
plt.xticks(rotation='vertical')
plt.title("Frequency of Subagency types", fontsize=15)
plt.show()


# In[56]:


fig = plt.figure(figsize=(18,6))
sns.countplot(x="YE", data = df)
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Year Established', fontsize=14)
plt.xticks(rotation='vertical')
plt.title("Year Established", fontsize=15)
plt.show()


# In[57]:


fig = plt.figure(figsize=(6,6))
sns.countplot(x="MGSTEM", data = df)
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Types', fontsize=14)
plt.title("Frequency of Mission-specific or General STEM?", fontsize=15)
plt.show()


# In[83]:


fig = plt.figure(figsize=(10,6))
sns.countplot(x="F1) Primary Investment Objective", data = df)
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Subagency', fontsize=14)
plt.title("Frequency of Agency types", fontsize=15)
plt.show()


# In[59]:


fig = plt.figure(figsize=(4,6))
sns.countplot(x="K) Eligibility Restrictions", data = df)
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Eligibility Restrictions', fontsize=14)
plt.title("Frequency of Eligibility Restrictions", fontsize=15)
plt.show()


# In[60]:


fig = plt.figure(figsize=(6,6))
sns.countplot(x="Q) Legislation Required to Shift Focus?", data = df)
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Legislation Required to Shift Focus', fontsize=14)
plt.title("Frequency of Legislation Required to Shift Focus", fontsize=15)
plt.show()


# In[61]:



funding_cols = ['FY2008','FY2009','FY2010']
df_new = df.drop(funding_cols,axis=1)
non_funding_variable=df_new.iloc[:,:8]
nonfunding_columnlist = non_funding_variable.columns.tolist()
for column in nonfunding_columnlist:
        score = mutual_info_score(df['target'],df[column])
        print("Mutual info score of the target variable and the",column," is :",score)


# ## STAGE 3:
# #1) Divide data into train & test samples. (70-30 split)
# #2) Select features & build xgboost model. You will be judged on roc_auc_score on test sample.

# In[69]:


def econding_cat_feature():
    lebelencoder = LabelEncoder()
    for column in df_new.columns:
        df_new[column]=lebelencoder.fit_transform(df_new[column])
    return df_new


# In[70]:


#econding the categorical features of the dataframe
df_new = econding_cat_feature()


# In[71]:


#task 1 of the stage 3
x = df_new.drop('target' ,axis=1)
y = df_new['target']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[80]:


from xgboost import XGBClassifier
clf = XGBClassifier(max_depth=7, n_estimators=300, learning_rate=0.05,min_child_weight=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# In[81]:


from sklearn.metrics import roc_auc_score
print("ROC Score : " + '{}'.format(roc_auc_score(y_test, y_pred)))


# In[82]:


from sklearn.metrics import accuracy_score
ac =accuracy_score(y_test,y_pred)
ac

