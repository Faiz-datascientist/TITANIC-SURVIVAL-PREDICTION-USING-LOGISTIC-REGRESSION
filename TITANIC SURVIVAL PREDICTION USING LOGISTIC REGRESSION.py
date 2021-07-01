#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df2=sns.load_dataset('titanic')


# In[3]:


df1=df2


# In[4]:


df1.head()


# In[5]:


df1.isna().sum()


# In[6]:


#checking the outliers
sns.boxplot(df1.age)


# In[13]:


#filling the missing values
df1['age']=df1['age'].fillna(df1['age'].median())


# In[14]:


df1.isna().sum()


# In[15]:


#drop the unwanted feature
df1=df1.drop("deck",axis=1)


# In[16]:


df1.isnull().sum()


# In[17]:


df1.dropna(inplace=True)


# In[18]:


df1.isnull().sum()


# In[19]:


#visualize the null values by using heatmap
sns.heatmap(df1.isnull())


# In[21]:


sns.boxplot(df1.survived,df1.age)


# In[22]:


sns.scatterplot(x=df1.survived,y=df1.age,hue=df1['sex'],style=df1['pclass'])


# In[23]:


df1.groupby("survived").std()


# In[24]:


df1.describe()


# In[26]:


#analysis:- we notice that pclass 1 is more likely to survive as compare to pclass 3
sns.countplot(x=df1['survived'],hue=df1.pclass)


# In[27]:


#Analysis:- we notice that female are thrice more likely to survive than male
sns.countplot(x=df1['survived'],hue=df1['sex'])


# In[28]:


#here in the graph i understand the correlation between the features 
#survival and fare correlation in 0.25 
sns.heatmap(df1.corr(),annot=True)


# In[29]:


sns.boxplot(x="survived", y="age",data=df1)
sns.stripplot(x="survived", y="age",data=df1,jitter=True ,edgecolor="black",hue=df1['sex'])
plt.title("SURVIVAL VS AGE",fontsize=20)


# In[30]:


df1['survived'].value_counts().plot(kind='hist')


# In[27]:


df['survived'].value_counts().plot(kind="pie",autopct='%.2f',figsize=(10,8))


# In[31]:


#Analysis:- we notice that more Age group peolpe are between 20-30 and very less 65-80
df1['age'].plot.hist(bins=5)


# In[32]:


#ANALYSIS:-we observe that most of the ticket bought under 100 and very few 200-500
df1['fare'].plot.hist(bins=5)


# In[ ]:





# In[33]:


#ANALYSIS:-we observe here in pclass more young people are travel
sns.barplot(df1['pclass'],df1['age'])


# In[35]:


#for better understanding we use distplot 
sns.distplot(df1[df1['survived']==0]['age'],hist=False)
sns.distplot(df1[df1['survived']==1]['age'],hist=False)


# In[37]:


pd.crosstab(df1['pclass'],df1['survived'])


# In[38]:


#ANALYSIS:-here we can esily vizualise pclass vs survived
sns.heatmap(pd.crosstab(df1['pclass'],df1['survived']))


# In[158]:


df1.head(5)


# In[173]:


#one-hot encoding
gender=pd.get_dummies(df1['sex'],drop_first=True)
gender.head()


# In[174]:


city=pd.get_dummies(df1['embarked'],drop_first=True)
city.head(5)


# In[181]:


pas_class=pd.get_dummies(df1['pclass'],drop_first=True)
pas_class.head()


# In[182]:


df2=pd.concat([df1,gender,city,pas_class],axis=1)
df2.head()


# In[183]:


df2.drop(['sex','pclass','embarked','parch','who','class','alone','embark_town','alive','adult_male'],axis=1,inplace=True)


# In[184]:


df2.head()


# # Training the model

# In[185]:


X=df2.drop(['survived'],axis=1)
y=df2['survived']


# In[186]:


X.head(5)


# In[187]:


y


# In[188]:


from sklearn.model_selection import train_test_split


# In[189]:


x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=42)


# In[190]:


print(" shape of x train",x_train.shape)
print(" shape of y train",y_train.shape)
print(" shape of x test",x_test.shape)
print(" shape of y test",y_test.shape)


# In[191]:


from sklearn.linear_model import LogisticRegression


# In[192]:


lr=LogisticRegression()


# In[193]:


lr.fit(x_train,y_train)


# In[195]:


prediction=lr.predict(x_test)


# In[196]:


from sklearn.metrics import classification_report


# In[197]:


classification_report(y_test,prediction)


# In[200]:


from sklearn.metrics import confusion_matrix


# In[201]:


confusion_matrix(y_test,prediction)


# In[202]:


from sklearn.metrics import accuracy_score


# In[203]:


accuracy_score(y_test,prediction)


# In[ ]:




