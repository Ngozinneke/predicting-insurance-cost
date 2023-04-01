#!/usr/bin/env python
# coding: utf-8

# # Medical insurance costs
# 
# This dataset was inspired by the book Machine Learning with R by Brett Lantz. 
# 
# The data contains medical information and costs billed by health insurance companies. 
# 
# It contains 1338 rows of data and the following columns:
# 
# age, gender, BMI, children, smoker, region and insurance charges.
# 
# ## Columns
# 
# age: age of primary beneficiary
# 
# sex: insurance contractor gender, female, male
# 
# bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
# objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
# 
# children: Number of children covered by health insurance / Number of dependents
# 
# smoker: Smoking
# 
# region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
# 
# charges: Individual medical costs billed by health insurance
# 
# ## Inspiration
# Can you accurately predict insurance costs?

# In[1]:


# Importing all required libraries for EDA
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("insurance.csv")


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.describe().T


# In[7]:


# Create a list to store all numerical variable
numerical_feature = [feature for feature in df.columns if df[feature].dtypes not in ['O', 'object'] ]

print("Number of Numerical Variable ", len(numerical_feature))

df[numerical_feature].head()


# In[8]:


# Create a list to store all Categorical variable
categorical_feature = [feature for feature in df.columns if df[feature].dtypes  in ['O', 'object'] ]

print("Number of Categorical Variable ", len(categorical_feature))

df[categorical_feature].head()


# In[9]:


# First of all analysing target feature
plt.figure(figsize=(14,5))
plt.plot(df["charges"])
plt.title("Plot for Charges")
plt.show()


# In[10]:


# Charges
plt.figure(figsize=(20,6))

plt.subplot(1,2,1)
plt.title('Distribution Plot')
sns.distplot(df["charges"])

plt.subplot(1,2,2)
plt.title('Box Plot')
sns.boxplot(y=df["charges"])

plt.show()


# In[11]:


# Children
plt.figure(figsize=(20,6))

plt.subplot(1,3,1)
plt.title("Counter Plot")
sns.countplot(x = 'children',data = df)

plt.subplot(1,3,2)
plt.title('Distribution Plot')
sns.distplot(df["children"])

plt.subplot(1,3,3)
plt.title('Box Plot')
sns.boxplot(y=df["children"])

plt.show()


# In[12]:


for feature in ['age', 'bmi']:
    plt.figure(figsize=(20,6))

    plt.subplot(1,3,1)
    plt.title("Counter Plot")
    sns.countplot(x = feature,data = df)

    plt.subplot(1,3,2)
    plt.title('Distribution Plot')
    sns.distplot(df[feature])

    plt.subplot(1,3,3)
    plt.title('Box Plot')
    sns.boxplot(y=df[feature])

    plt.show()


# In[13]:


categorical_feature


# In[14]:


plt.figure(figsize=(26,6))

plt.subplot(1,3,1)
plt.title("Counter Plot For Sex")
sns.countplot(x = "sex", data = df)

plt.subplot(1,3,2)
plt.title("Counter Plot For Smoker")
sns.countplot(x = "smoker", data = df)

plt.subplot(1,3,3)
plt.title("Counter Plot For Region")
sns.countplot(x = "region", data = df)

plt.show()


# In[15]:


plt.figure(figsize=(26,5))

plt.subplot(1,4,1)
plt.title("Scatter Plot between age and charges")
sns.scatterplot(x=df['age'], y=df['charges'])

plt.subplot(1,4,2)
plt.title("Scatter Plot between bmi and charges")
sns.scatterplot(x=df['bmi'], y=df['charges'])

plt.subplot(1,4,3)
plt.title("Scatter Plot between children and charges")
sns.scatterplot(x=df['children'], y=df['charges'])

plt.subplot(1,4,4)
plt.title("Scatter Plot between charges and charges")
sns.scatterplot(x=df['charges'], y=df['charges'])

plt.show()


# In[16]:


for feature in categorical_feature:
    plt.figure(figsize=(26,5))
    plt.subplot(1,4,1)
    plt.title("Scatter Plot between age and charges")
    sns.scatterplot(x=df['age'], y=df['charges'], hue=df[feature])

    plt.subplot(1,4,2)
    plt.title("Scatter Plot between bmi and charges")
    sns.scatterplot(x=df['bmi'], y=df['charges'], hue=df[feature])

    plt.subplot(1,4,3)
    plt.title("Scatter Plot between children and charges")
    sns.scatterplot(x=df['children'], y=df['charges'], hue=df[feature])

    plt.subplot(1,4,4)
    plt.title("Scatter Plot between charges and charges")
    sns.scatterplot(x=df['charges'], y=df['charges'], hue=df[feature])

    plt.show()
    print("="*210)


# In[17]:


dataset = pd.get_dummies(df)
dataset.head()


# In[18]:


dataset = dataset.drop(["sex_female","smoker_no"], axis=1)


# In[19]:


dataset.head()


# In[20]:



dataset.rename(columns = {'sex_male':'sex'}, inplace = True)
dataset.rename(columns = {'smoker_yes':'smoker'}, inplace = True)


# In[21]:


y = dataset["charges"]
X = dataset.drop(["charges"], axis=1)


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[24]:


#Buidlign a Random Forest Regression Model
rf_model = RandomForestRegressor(n_estimators = 100)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)


# In[25]:


# Model Evulation
r2 = r2_score(rf_pred, y_test)
print(f"The R2 For Our Linear Regression Model is {r2}")


# In[27]:


# Predicting
#predicted = rf_model.predict([[32, 23.5,1,1, 0]])
#print('Predicted Chareg for given patient = ', predicted)


# In[28]:


pip install shap


# In[29]:


import shap


# In[30]:


shap_values = shap.TreeExplainer(rf_model).shap_values(X_test)
shap.summary_plot(shap_values,X_test,plot_type="bar")


# In[31]:


shap.summary_plot(shap_values, X_test)


# In[32]:


shap.initjs()
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values, X_test.iloc[0,:])


# In[33]:


pip install explainerdashboard


# In[34]:


import explainerdashboard


# In[35]:


from explainerdashboard import RegressionExplainer, ExplainerDashboard
explainer = RegressionExplainer(rf_model, X_test, y_test)


# In[ ]:


ExplainerDashboard(explainer).run()

