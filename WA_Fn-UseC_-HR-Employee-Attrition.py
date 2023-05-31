#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import sklearn as sk
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder , StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split , cross_val_score ,GridSearchCV
from sklearn.ensemble import RandomForestClassifier , ExtraTreesClassifier , GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn.metrics import accuracy_score ,classification_report , confusion_matrix,roc_curve , roc_curve , roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib


# In[2]:


df=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")


# In[3]:


df


# In[4]:


df.shape


# There are 1470 rows and 35 columns present in the dataset.34 feature columns and 1 target variable

# In[5]:


df.columns


# In[6]:


df.isnull().sum()


# There is no missing values present in the data.

# In[7]:


df.info()


# Two Types of data type present in the dataset Int64 and Object

# In[8]:


sns.heatmap(df.isnull())


# In[9]:


df["Attrition"].value_counts()


# In[10]:


for i in df.columns:
    print(df[i].value_counts())
    print("/n")


# In[11]:


df.nunique()


# As we can see EmployeeCount Column has one unique value and Employee number column has all unique values ... so we won't get any changes in the data set if we remove these columns.

# In[12]:


df=df.drop(["EmployeeCount","EmployeeNumber"],axis=1)


# In[13]:


df.shape


# In[14]:


# Univariate analysis


# In[15]:


sns.countplot(x="Attrition",data=df)


# "No" Attrition rate is much high than  "Yes"

# In[16]:


sns.countplot(x="BusinessTravel" , data=df)


# As We can see Non-Travel count very less as compare to Travel_rarely and Travel_Frequently

# In[17]:


sns.countplot(x= "Department" , data=df)


# High volume are present in Reasearch & Development Department.

# In[18]:


sns.countplot(x="EducationField" , data=df)


# Highest number of employees are from Lifescience EnducationField

# In[19]:


sns.countplot(x="Gender",data=df)


# Male employees are high as compare to female.

# In[20]:


sns.countplot(x="JobRole",data=df)
df["JobRole"].value_counts()
plt.subplots_adjust(hspace=1.0)
plt.xticks(rotation=90)


# High numbers of Sales Executive roles are present in the data.

# In[21]:


sns.countplot(x="MaritalStatus" , data=df)


# more married employees are there as compare to single and Divorced

# In[22]:


sns.countplot(x="OverTime" , data=df)


# In[23]:


# Bivariate


# In[24]:


numerical_col=df.select_dtypes(include=[np.number]).columns
numerical_col


# In[25]:


numerical_col.value_counts().sum()


# In[26]:


categorical_col=df.select_dtypes(exclude=[np.number]).columns
categorical_col


# In[27]:


plt.figure(figsize=(10,6) , facecolor= "white")
pn=1
for col in numerical_col:
    if pn<=24:
        ax=plt.subplot(4,6,pn)
        sns.distplot(df[col] , color="m")
        plt.xlabel(col,fontsize=12)
        plt.ylabel(col,fontsize=10)
    pn+=1
plt.tight_layout()


# # Bivariate analysis

# In[28]:


clm=['Age', 'DailyRate', 'DistanceFromHome', 'Education',
       'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
       'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
       'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']


# In[29]:


clm


# In[30]:


fig=plt.subplots(figsize=(40,40))
for p,q in enumerate(clm):
    plt.subplot(6,4,p+1)
    plt.subplots_adjust(hspace=1.0)
    sns.countplot(x=q,data=df,hue="Attrition")
    plt.xticks(rotation=90)


# In[31]:


df.describe()


# As we can see there is a high difference between max and Q3 in TotalWorkingYears & 	MonthlyIncome

# In[32]:


fig = plt.figure(figsize=(10,10))

fig , ax = plt.subplots(6,3,figsize=(10,10))
for v , subplot in zip(numerical_col , ax.flatten()):
    sns.boxplot(x=df[v] , ax= subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# As we can see outliers are present in oTtalWorkingYears & 	MonthlyIncome so outliers are present in these columns .

# Lets remove outliers

# In[33]:


outlier = df[["TotalWorkingYears","MonthlyIncome"]]


# In[34]:


Q1=outlier.quantile(0.25)
Q3=outlier.quantile(0.75)
IQR=Q3-Q1


# In[35]:


df1=df[~((df<(Q1-1.5*IQR))|(df>(Q3+1.5*IQR))).any(axis=1)]


# In[36]:


df1.shape


# Lets check loss%

# In[37]:


print(" The data loss percentage :- " , ((df.shape[0]-df1.shape[0])/df.shape[0])*100)


# As the data loss percentage is less than 10 so we can go with this mathod

# Lets check the correlation

# In[38]:


df=df1


# In[39]:


df.duplicated().sum()


# In[40]:


# hence there is no duplicate data present in the data set


# # Encoding Categorical col into numerical

# In[41]:


OE=OrdinalEncoder()
for i in df.columns:
    if df[i].dtypes=="object":
        df[i]=OE.fit_transform(df[i].values.reshape(-1,1))
df


# In[42]:


cor=df.corr()


# In[43]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr())


# In[44]:


cor["Attrition"].sort_values(ascending=False)


# As We can see all the employees are above 18 so we can remove the column

# In[45]:


df.drop("Over18",axis=1)


# # Feature Scalling using Standard Scalarization

# In[46]:


x=df.drop(["Attrition","Over18"],axis=1)
y=df["Attrition"]


# In[47]:


scaler=StandardScaler()
x=pd.DataFrame(scaler.fit_transform(x),columns=x.columns)


# In[48]:


x


# In[49]:


y.shape


# # Checking variance Inflation Factor(VIF)

# In[50]:


VIF=pd.DataFrame()
VIF["VIF Values"]=[variance_inflation_factor(x.values,i) for i in range(len(x.columns))]
VIF["Features"]=x.columns
VIF


# Here we can see VIF value of all the columns are less than 10.

# # Modelling

# Finding the bestRandom State

# In[51]:


maxAccu=0
maxRS=0
for i in range(1,300):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=i)
    RFR=RandomForestClassifier()
    RFR.fit(x_train,y_train)
    pred=RFR.predict(x_test)
    accu=accuracy_score(y_test,pred)
    if accu>maxAccu:
        maxAccu=accu
        maxRS=i
print("Best accuracy Score is :-",maxAccu , "and random state is ", maxRS)


# # Classification Algorithm

# RandomForestClassifier

# In[52]:


RFC=RandomForestClassifier()
RFC.fit(x_train,y_train)
pred_RFC=RFC.predict(x_test)
print(accuracy_score(y_test,pred_RFC))
print(confusion_matrix(y_test,pred_RFC))
print(classification_report(y_test,pred_RFC))


# The Accuracy for ithis model is 83%

# LogisticRegression

# In[53]:


LR=LogisticRegression()
LR.fit(x_train,y_train)
pred_LR=LR.predict(x_test)
print(accuracy_score(y_test,pred_LR))
print(confusion_matrix(y_test,pred_LR))
print(classification_report(y_test,pred_LR))


# The Accuracy score using logistic regression is 86%

# Support vactor Machine Classifier

# In[54]:


svm=SVC()
svm.fit(x_train,y_train)
pred_svm=svm.predict(x_test)
print(accuracy_score(y_test,pred_svm))
print(confusion_matrix(y_test,pred_svm))
print(classification_report(y_test,pred_svm))


# The Accuracy score using SVC is 85%

# GradientBoosting Classfication

# In[55]:


GB=GradientBoostingClassifier()
GB.fit(x_train,y_train)
pred_GB=GB.predict(x_test)
print(accuracy_score(y_test,pred_GB))
print(confusion_matrix(y_test,pred_GB))
print(classification_report(y_test,pred_GB))


# The accuracy score using GradientBoostingClassifier is 83%

# Adaboostingclassifier

# In[56]:


AB=AdaBoostClassifier()
AB.fit(x_train,y_train)
pred_AB=AB.predict(x_test)
print(accuracy_score(y_test,pred_AB))
print(confusion_matrix(y_test,pred_AB))
print(classification_report(y_test,pred_AB))


# The Accuracy score using Adaboost Classifier is 82%

# ExtaTreeClassifier

# In[57]:


ETC=ExtraTreesClassifier()
ETC.fit(x_train,y_train)
pred_ETC=ETC.predict(x_test)
print(accuracy_score(y_test,pred_ETC))
print(confusion_matrix(y_test,pred_ETC))
print(classification_report(y_test,pred_ETC))


# The accuracy score using Extratree classifier is 85%

# BaggingClassifier

# In[58]:


BC=BaggingClassifier()
BC.fit(x_train,y_train)
pred_BC=BC.predict(x_test)
print(accuracy_score(y_test,pred_BC))
print(confusion_matrix(y_test,pred_BC))
print(classification_report(y_test,pred_BC))


# The Accuracy score using the baggingClassifier is 81%

# # Cross validation score

# In[59]:


# Checing cv score for Random forest classifier
score=cross_val_score(RFC , x,y)
print(score)
print(score.mean())
print("The difference between the Accuracy score and the cross validation score is ",accuracy_score(y_test,pred_RFC)-score.mean())


# In[60]:


# checking cv for Logistic Regression
score=cross_val_score(LR , x,y)
print(score)
print(score.mean())
print("The difference between the Accuracy score and the cross validation score is ",accuracy_score(y_test,pred_LR)-score.mean())


# In[61]:


# checking cv for SVC
score=cross_val_score(svm, x,y)
print(score)
print(score.mean())
print("The difference between the Accuracy score and the cross validation score is ",accuracy_score(y_test,pred_svm)-score.mean())


# In[62]:


# checking cv for GradientBoosting classifier
score=cross_val_score(GB , x,y)
print(score)
print(score.mean())
print("The difference between the Accuracy score and the cross validation score is ",accuracy_score(y_test,pred_GB)-score.mean())


# In[63]:


# Checing cv for ExtraTreesclassifier
score=cross_val_score(ETC , x,y)
print(score)
print(score.mean())
print("The difference between the Accuracy score and the cross validation score is ",accuracy_score(y_test,pred_ETC)-score.mean())


# In[64]:


# Checing cv for AdaBoostClassifier
score=cross_val_score(AB , x,y)
print(score)
print(score.mean())
print("The difference between the Accuracy score and the cross validation score is ",accuracy_score(y_test,pred_AB)-score.mean())


# In[65]:


# checking cv for BaggigClassifier
score=cross_val_score(BC , x,y)
print(score)
print(score.mean())
print("The difference between the Accuracy score and the cross validation score is ",accuracy_score(y_test,pred_BC)-score.mean())


# ExtraTrees classifier is the best model as the difference between the accuracy score and cross validation score is least

# # Hyper parameter tuning

# In[66]:


parameters={'criterion' :['gini','entropy'],'random_state':[10,50,100],'max_depth':[0,10,20],'n_jobs' : [-2,-1,1],'n_estimators':[50,100,200,300]}


# In[67]:


GCV=GridSearchCV(ExtraTreesClassifier(),parameters,cv=5)


# In[68]:


final_model=ExtraTreesClassifier(criterion="entropy",max_depth=20,n_estimators=200,n_jobs=-2,random_state=10)


# In[69]:


final_model.fit(x_train,y_train)
pred_final_model=final_model.predict(x_test)
accu=accuracy_score(y_test,pred_final_model)
print(accu*100)


# # Saving the model

# In[70]:


joblib.dump(final_model,"HR-Employee-Attrition")


# In[71]:


model=joblib.load("HR-Employee-Attrition")


# In[72]:


prediction=model.predict(x_test)
prediction


# In[73]:


r=np.array(y_test)
data=pd.DataFrame()
data["Original"]=r
data["Prediction"]=prediction
data

