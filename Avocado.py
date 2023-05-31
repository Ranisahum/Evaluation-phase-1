#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import OneHotEncoder , LabelEncoder,OrdinalEncoder,StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split , cross_val_score ,GridSearchCV
from sklearn.ensemble import RandomForestClassifier , ExtraTreesClassifier , GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn.metrics import accuracy_score ,classification_report , confusion_matrix,roc_curve , roc_curve , roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
from sklearn.linear_model import LinearRegression ,Lasso,Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor , RandomForestRegressor


# In[2]:


df=pd.read_csv("avocado.csv")


# In[3]:


df.shape


# There are 18249 rows and 14 columns present in the data set

# In[4]:


df.columns


# There are two target variable present in the data set

# In[5]:


df


# In[6]:


df.isnull().sum()


# As we can see there is no null values present in the data set

# In[7]:


df["Unnamed: 0"].value_counts


# As we can see unnamed columns only contais index number so if we remove the same from data set it won't effect

# In[8]:


df=df.drop("Unnamed: 0",axis=1)


# In[9]:


df.info()


# Three types of datatyppe present in the data set i.e int,float,object

# Converting Data type of Date from object to date

# In[10]:


df["Date"]=pd.to_datetime(df["Date"])


# In[11]:


df.info()


# Now the data type of date has been changed to date

# In[12]:


df.duplicated().sum()


# There is no duplicate value present in the data set

# # Lets get the numerical anf categorical columns separately

# In[13]:


numerical_col=df.select_dtypes(include=['int64','float64','datetime64[ns]']).columns


# In[14]:


numerical_col


# In[15]:


categorical_col=df.select_dtypes(include=['object']).columns


# In[16]:


categorical_col


# # univariate analysis

# In[17]:


sns.countplot(x="type",data=df)


# As we can see equal number of avocado types present in the data

# In[18]:


sns.countplot(x='region', data = df)
plt.xticks(rotation=90)


# In[19]:


df["region"].value_counts


# In[20]:


df.describe()


# Checking average prices over years

# In[21]:


sns.boxplot(x="year", y="AveragePrice",data=df).figsize=(80,15)


# As we can notice price was high in 2017

# checking average price across the region

# In[22]:


df.groupby("region")["AveragePrice"].sum().sort_values(ascending=False).plot(kind="bar",figsize=(15,5))


# HartfordSpringfield being highest and Houston being lowest price to get cheap Avocado.

# # Checking skewness

# In[23]:


df.skew()


# As we can see maany of the columns are having skewness , lets remove skewness 

# In[24]:


df["4770"] = np.cbrt(df["4770"])
df["XLarge Bags"] = np.cbrt(df["XLarge Bags"])


# In[25]:


df.skew()


# In[26]:


df["Total Bags"] = np.cbrt(df["Total Bags"])


# In[27]:


df.skew()


# In[28]:


df["Large Bags"] = np.cbrt(df["Large Bags"])


# In[29]:


df.skew()


# In[30]:


df["Small Bags"] = np.cbrt(df["Small Bags"])


# In[31]:


df.skew()


# In[32]:


df["4225"] = np.cbrt(df["4225"])
df["4046"] = np.cbrt(df["4046"])


# In[33]:


df.skew()


# In[34]:


df["Total Volume"] = np.cbrt(df["Total Volume"])


# In[35]:


df.skew()


# We have removed skewness from all of the columns.

# In[36]:


df.corr()


# In[37]:


plt.figure(figsize=(26,14))
sns.heatmap(df.corr() ,annot=True , fmt='0.2f' , linewidth=0.2, linecolor='black',cmap="Spectral")
plt.xlabel("Figure", fontsize=14)
plt.ylabel("Features_Name" , fontsize=14)
plt.title("Descriptive Graph" , fontsize=20)


# In[38]:


df.corr()["AveragePrice"].sort_values(ascending=True)


# In[39]:


# encoding the object to numerical


# In[40]:


OE=OrdinalEncoder()
for i in df.columns:
    if df[i].dtypes=="object":
        df[i]=OE.fit_transform(df[i].values.reshape(-1,1))
df


# # Feature Scalling using Standard Scalarization

# In[41]:


x=df.drop(["region","Date"],axis=1)
y=df["region"]


# In[42]:


y.shape


# # Feature Scalling using Standard Scalarization

# In[43]:


scaler=StandardScaler()
x=pd.DataFrame(scaler.fit_transform(x),columns=x.columns)


# In[44]:


x


# # Checking variance Inflation Factor(VIF)
# 

# In[45]:


VIF=pd.DataFrame()
VIF["VIF Values"]=[variance_inflation_factor(x.values,i) for i in range(len(x.columns))]
VIF["Features"]=x.columns
VIF


# As We can see 6 columns are having VIF value more than 10. so we are going to remove Total volume column As VIF value is too high in this column

# In[46]:


x.drop("Total Volume" , axis=1, inplace=True)


# In[47]:


VIF=pd.DataFrame()
VIF["VIF Values"]=[variance_inflation_factor(x.values,i) for i in range(len(x.columns))]
VIF["Features"]=x.columns
VIF


# In[48]:


x.drop("Total Bags" , axis=1, inplace=True)


# In[49]:


VIF=pd.DataFrame()
VIF["VIF Values"]=[variance_inflation_factor(x.values,i) for i in range(len(x.columns))]
VIF["Features"]=x.columns
VIF


# Now the vif value is less than 10 in each of the columns

# # Modelling

# Finding the Best Random State

# In[50]:


maxAccu=0
maxRS=0
for i in range(1,300):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=i)
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

# In[51]:


RFC=RandomForestClassifier()
RFC.fit(x_train,y_train)
pred_RFC=RFC.predict(x_test)
print(accuracy_score(y_test,pred_RFC))
print(confusion_matrix(y_test,pred_RFC))
print(classification_report(y_test,pred_RFC))


# The Accuracy for ithis model is 91%

# LogisticRegression

# In[52]:


LR=LogisticRegression()
LR.fit(x_train,y_train)
pred_LR=LR.predict(x_test)
print(accuracy_score(y_test,pred_LR))
print(confusion_matrix(y_test,pred_LR))
print(classification_report(y_test,pred_LR))


# The Accuracy score using logistic regression is 86%

# Support vactor Machine Classifier

# In[53]:


svm=SVC()
svm.fit(x_train,y_train)
pred_svm=svm.predict(x_test)
print(accuracy_score(y_test,pred_svm))
print(confusion_matrix(y_test,pred_svm))
print(classification_report(y_test,pred_svm))


# The Accuracy score using SVC is 70%

# Adaboostingclassifier

# In[54]:


AB=AdaBoostClassifier()
AB.fit(x_train,y_train)
pred_AB=AB.predict(x_test)
print(accuracy_score(y_test,pred_AB))
print(confusion_matrix(y_test,pred_AB))
print(classification_report(y_test,pred_AB))


# The Accuracy score using Adaboost Classifier is 6%

# ExtaTreeClassifier

# In[55]:


ETC=ExtraTreesClassifier()
ETC.fit(x_train,y_train)
pred_ETC=ETC.predict(x_test)
print(accuracy_score(y_test,pred_ETC))
print(confusion_matrix(y_test,pred_ETC))
print(classification_report(y_test,pred_ETC))


# The accuracy score using Extratree classifier is 93%

# BaggingClassifier

# In[58]:


BC=BaggingClassifier()
BC.fit(x_train,y_train)
pred_BC=BC.predict(x_test)
print(accuracy_score(y_test,pred_BC))
print(confusion_matrix(y_test,pred_BC))
print(classification_report(y_test,pred_BC))


# The Accuracy score using the baggingClassifier is 86%

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


# Checing cv for ExtraTreesclassifier
score=cross_val_score(ETC , x,y)
print(score)
print(score.mean())
print("The difference between the Accuracy score and the cross validation score is ",accuracy_score(y_test,pred_ETC)-score.mean())


# In[63]:


# Checing cv for AdaBoostClassifier
score=cross_val_score(AB , x,y)
print(score)
print(score.mean())
print("The difference between the Accuracy score and the cross validation score is ",accuracy_score(y_test,pred_AB)-score.mean())


# In[64]:


# checking cv for BaggigClassifier
score=cross_val_score(BC , x,y)
print(score)
print(score.mean())
print("The difference between the Accuracy score and the cross validation score is ",accuracy_score(y_test,pred_BC)-score.mean())


# # Hyper parameter tuning
# 

# In[65]:


parameters={'criterion' :['gini','entropy'],'random_state':[10,50,100],'max_depth':[0,10,20],'n_jobs' : [-2,-1,1],'n_estimators':[50,100,200,300]}


# In[66]:


GCV=GridSearchCV(ExtraTreesClassifier(),parameters,cv=5)


# In[67]:


final_model=ExtraTreesClassifier(criterion="entropy",max_depth=20,n_estimators=200,n_jobs=-2,random_state=10)


# In[68]:


final_model.fit(x_train,y_train)
pred_final_model=final_model.predict(x_test)
accu=accuracy_score(y_test,pred_final_model)
print(accu*100)


# # Saving the model

# In[69]:


joblib.dump(final_model,"Avocado_region")


# In[70]:


model=joblib.load("Avocado_region")


# In[71]:


prediction=model.predict(x_test)
prediction


# In[72]:


r=np.array(y_test)
data=pd.DataFrame()
data["Original"]=r
data["Prediction"]=prediction
data


# # Average price predication

# In[73]:


df


# In[74]:


df.drop("Date",axis=1)


# In[75]:


X=df.drop(["AveragePrice","Date"],axis=1)
Y=df["AveragePrice"]


# Feature Scaling using Standard Scalarization

# In[76]:


scaler= StandardScaler()
X=pd.DataFrame(scaler.fit_transform(X),columns=X.columns)


# In[77]:


X


# checking VIF (Variance Inflation Factor)

# In[78]:


vif=pd.DataFrame()
vif["VIF Values"]= [variance_inflation_factor(X.values,i) for i in range(len(X.columns))]
vif["Features"]= X.columns
vif


# As we can see 6 columns are having VIF value more than 10

# In[79]:


X.drop("Total Volume", axis=1 , inplace=True)


# In[80]:


# again checking vif value
vif=pd.DataFrame()
vif["VIF Value"]=[variance_inflation_factor(X.values,i)for i in range(len(X.columns))]
vif["Independent variable"]=X.columns
vif


# In[81]:


X.drop("Total Bags", axis=1 , inplace=True)


# In[82]:


# again checking vif value
vif=pd.DataFrame()
vif["VIF Value"]=[variance_inflation_factor(X.values,i)for i in range(len(X.columns))]
vif["Independent variable"]=X.columns
vif


# Now the VIF value lies under 10

# # Modelling

# In[83]:


maxaccu=0
maxRS=0
for i in range(1,300):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.20,random_state=i)
    lr=LinearRegression()
    lr.fit(X_train,Y_train)
    pred=lr.predict(X_test)
    acc=r2_score(Y_test,pred)
    if acc>maxaccu:
        maxaccu=acc
        maxRS=i
print("The Maximun r2 score is ",maxaccu, "on random State",maxRS)
    


# In[84]:


# RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(X_train,Y_train)
pred_rf=rf.predict(X_test)
pred_train=rf.predict(X_train)
print("R2 Score  : ",r2_score(Y_test,pred_rf))
print("R2_score on training data :",r2_score(Y_train,pred_train)*100)
print("Mean Absolute Error : ", mean_absolute_error(Y_test,pred_rf))
print("Mean Squared Error : ",mean_squared_error(Y_test,pred_rf))
print("Root Mean Squared Error",np.sqrt(mean_squared_error(Y_test,pred_rf)))


# In[85]:


# LinearRegression Model
lr=LinearRegression()
lr.fit(X_train,Y_train)
pred_lr=lr.predict(X_test)
pred_train=lr.predict(X_train)
print("R2 Score  : ",r2_score(Y_test,pred_lr))
print("R2_score on training data :",r2_score(Y_train,pred_train)*100)
print("Mean Absolute Error : ", mean_absolute_error(Y_test,pred_lr))
print("Mean Squared Error : ",mean_squared_error(Y_test,pred_lr))
print("Root Mean Squared Error",np.sqrt(mean_squared_error(Y_test,pred_lr)))


# In[86]:


# GradientBoostingRegressor Model
GBR=GradientBoostingRegressor()
GBR.fit(X_train,Y_train)
pred_GBR=GBR.predict(X_test)
pred_train=GBR.predict(X_train)
print("R2 Score  : ",r2_score(Y_test,pred_GBR))
print("R2_score on training data :",r2_score(Y_train,pred_train)*100)
print("Mean Absolute Error : ", mean_absolute_error(Y_test,pred_GBR))
print("Mean Squared Error : ",mean_squared_error(Y_test,pred_GBR))
print("Root Mean Squared Error",np.sqrt(mean_squared_error(Y_test,pred_GBR)))


# In[87]:


#KNeighborsRegressor Model
from sklearn.neighbors import KNeighborsRegressor as KNN
knn=KNN()
knn.fit(X_train,Y_train)
pred_knn=knn.predict(X_test)
pred_train=GBR.predict(X_train)
print("R2 Score  : ",r2_score(Y_test,pred_knn))
print("R2_score on training data :",r2_score(Y_train,pred_train)*100)
print("Mean Absolute Error : ", mean_absolute_error(Y_test,pred_knn))
print("Mean Squared Error : ",mean_squared_error(Y_test,pred_knn))
print("Root Mean Squared Error",np.sqrt(mean_squared_error(Y_test,pred_knn)))


# In[88]:


# Lasso Regression
lasso=Lasso()
lasso.fit(X_train,Y_train)
pred_lasso = lasso.predict(X_test)
pred_train= lasso.predict(X_train)
print('R2_score :',r2_score(Y_test,pred_lasso))
print('R2_score on training Data :', r2_score(Y_train,pred_train)*100)
print('Mean Absolute Error : ', mean_absolute_error(Y_test,pred_lasso))
print('Mean Squared error :-',mean_squared_error(Y_test,pred_lasso))
print("Root mean squared error :-",np.sqrt(mean_squared_error(Y_test,pred_lasso)))


# In[89]:


# Ridge Regression
rd=Ridge()
rd.fit(X_train,Y_train)
pred_rd = rd.predict(X_test)
pred_train= rd.predict(X_train)
print('R2_score :',r2_score(Y_test,pred_rd))
print('R2_score on training Data :', r2_score(Y_train,pred_train)*100)
print('Mean Absolute Error : ', mean_absolute_error(Y_test,pred_rd))
print('Mean Squared error :-',mean_squared_error(Y_test,pred_rd))
print("Root mean squared error :-",np.sqrt(mean_squared_error(Y_test,pred_rd)))


# In[90]:


#DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(X_train,Y_train)
pred_dtr = dtr.predict(X_test)
pred_train= dtr.predict(X_train)
print('R2_score :',r2_score(Y_test,pred_dtr))
print('R2_score on training Data :', r2_score(Y_train,pred_train)*100)
print('Mean Absolute Error : ', mean_absolute_error(Y_test,pred_dtr))
print('Mean Squared error :-',mean_squared_error(Y_test,pred_dtr))
print("Root mean squared error :-",np.sqrt(mean_squared_error(Y_test,pred_dtr)))


# In[91]:


#SupportVectorRegressor
from sklearn.svm import SVR
svr=SVR()
dtr.fit(X_train,Y_train)
pred_svr= dtr.predict(X_test)
pred_train= dtr.predict(X_train)
print('R2_score :',r2_score(Y_test,pred_svr))
print('R2_score on training Data :', r2_score(Y_train,pred_train)*100)
print('Mean Absolute Error : ', mean_absolute_error(Y_test,pred_svr))
print('Mean Squared error :-',mean_squared_error(Y_test,pred_svr))
print("Root mean squared error :-",np.sqrt(mean_squared_error(Y_test,pred_svr)))


# In[92]:


#ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
rtr=ExtraTreesRegressor()
dtr.fit(X_train,Y_train)
pred_rtr= dtr.predict(X_test)
pred_train= dtr.predict(X_train)
print('R2_score :',r2_score(Y_test,pred_rtr))
print('R2_score on training Data :', r2_score(Y_train,pred_train)*100)
print('Mean Absolute Error : ', mean_absolute_error(Y_test,pred_rtr))
print('Mean Squared error :-',mean_squared_error(Y_test,pred_rtr))
print("Root mean squared error :-",np.sqrt(mean_squared_error(Y_test,pred_rtr)))


# # lets check Cross val score

# In[93]:


score=cross_val_score(lr,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_lr)-score.mean())*100)


# In[94]:


score=cross_val_score(rf,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_rf)-score.mean())*100)


# In[95]:


score=cross_val_score(knn,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_knn)-score.mean())*100)


# In[96]:


score=cross_val_score(GBR,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_GBR)-score.mean())*100)


# In[97]:


score=cross_val_score(lasso,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_lasso)-score.mean())*100)


# In[98]:


score=cross_val_score(rd,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_rd)-score.mean())*100)


# In[99]:


score=cross_val_score(dtr,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_dtr)-score.mean())*100)


# In[100]:


score=cross_val_score(svr,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_svr)-score.mean())*100)


# In[101]:


score=cross_val_score(rtr,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_rtr)-score.mean())*100)


# In[102]:


GBR.set_params(learning_rate=0.1, n_estimators=200, 
subsample=0.8, max_depth=5,
max_features='sqrt', min_samples_leaf=4, min_samples_split=10)


# In[103]:


GBR.fit(X_train,Y_train)


# In[104]:


y_pred_train = GBR.predict(X_train)
y_pred_test = GBR.predict(X_test)


# In[105]:


y_pred_train


# In[106]:


y_pred_test

