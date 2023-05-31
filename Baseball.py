#!/usr/bin/env python
# coding: utf-8

# # BASEBALL PROJECT

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import numpy as np
import sklearn as sk
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor , RandomForestRegressor
from sklearn.linear_model import LinearRegression ,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# In[4]:


df=pd.read_csv("baseball.csv")


# In[5]:


df.head()


# In[6]:


df


# # EDA(Exploratory Data Analysis) analysis

# In[7]:


df.shape


# In[8]:


df.columns


# There are 30 rows with 16 feature columns and 1 target variable

# In[9]:


df.info()


# There are two types of datatype present in the data set i.e int and float, We can see there is no null values present in the data set

# In[10]:


df.isnull().sum()


# As we can see there is no null values present in the data

# In[11]:


df.duplicated()


# As we can observe there is no duplicate value present in the data set.

# In[12]:


df.nunique().to_frame("Number of unique value")


# In[13]:


for i in df.columns:
    print(df[i].value_counts())
    print("\n")


# As per above explanation we can observe most of the columns are having high unique value

# In[14]:


df.describe()


# We can see there is a huge difference between the max and Q3 in R,SV & E columns, hence outliers are present in the data set.there is less difference between mean and median so normal disctribution occurs.Lets do data visualzation to check the same

# In[15]:


sns.distplot(df["ERA"],color='PURPLE')


# In[16]:


sns.distplot(df["R"],color='purple')


# Above graph represents that R column is a binomial disrtibution

# In[17]:


sns.distplot(df["AB"],color='purple')


# In[18]:


sns.distplot(df["H"],color='purple')


# In[19]:


sns.distplot(df["2B"],color='red')


# In[20]:


sns.distplot(df["3B"],color='purple')


# In[21]:


sns.distplot(df["HR"],color='green')


# In[22]:


sns.distplot(df["BB"],color='BLUE')


# In[23]:


sns.distplot(df["SO"],color='GREY')


# In[24]:


sns.distplot(df["SB"],color='PINK')


# In[25]:


sns.distplot(df["RA"],color='PURPLE')


# In[26]:


sns.distplot(df["ER"],color='GREEN')


# In[27]:


sns.distplot(df["ERA"],color='PURPLE')


# In[28]:


sns.distplot(df["CG"],color='GREEN')


# In[29]:


sns.distplot(df["SHO"],color='PURPLE')


# In[30]:


sns.distplot(df["SV"],color='GREEN')


# In[31]:


sns.distplot(df["E"],color='PURPLE')


# LETS DO THE VISUALIZATION IN A SINGLE GRAPH

# In[32]:


df1= df.melt(var_name='cols',  value_name='vals')


# In[33]:


sns.displot(kind='kde', data=df1, col='cols', col_wrap=4, x='vals', hue="cols", facet_kws={'sharey': False, 'sharex': False})


# As per the visualization few columns(W,AB,2B,3B,BB,SO,SB,RA,ER,ERA,SV)has normation distribution and few columns(R,H,CG,E,SHO) has binomial distribution so there is no skewness present in the data.

# checking for outliers

# In[34]:


numerical_col=df.select_dtypes(include=[np.number]).columns


# In[35]:


fig = plt.figure(figsize=(10,10))

fig , ax = plt.subplots(6,3,figsize=(10,10))
for v , subplot in zip(numerical_col , ax.flatten()):
    sns.boxplot(x=df[v] , ax= subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)


# In[36]:


# Removing outliers
outlier=df[["E","SV","R","SHO","ERA"]]
Z=np.abs(zscore(outlier))
Z


# In[37]:


np.where(Z>3)


# In[38]:


df1=df[(Z<3).all(axis=1)]
df1.shape


# In[39]:


#let's check data loss ppercentage
print("Data loss percentage :- ",((df.shape[0]-df1.shape[0])/df.shape[0])*100)


# Data loss percentage is 3.33% so we can go with the Zscore mathod

# In[40]:


Q1=outlier.quantile(0.25)
Q3=outlier.quantile(0.75)
IQR=Q3-Q1


# In[41]:


df2=df1[~((df1 < (Q1-1.5*IQR)) |(df1>(Q3+1.5*IQR))).any(axis=1)]


# In[42]:


df2.shape


# In[43]:


print("Data loss percentage after removing the outliers with IQR method is :-",((df1.shape[0]-df2.shape[0])/df1.shape[0])*100)


# As we can see data loss percentage is more than 30% which is not acceptable we can't go ahead with IQR method

# Lets check the corelation between each column

# In[44]:


df=df1


# In[45]:


df.corr()


# In[46]:


df.corr()["W"].sort_values(ascending = True)


# There are positive and negative corelation exsting between the cols

# Separating Features and Terget variable

# In[47]:


x=df.drop("W",axis=1)
Y=df["W"]


# Feature Scaling using Standard Scalarization

# In[48]:


scaler= StandardScaler()
X=pd.DataFrame(scaler.fit_transform(x),columns=x.columns)


# In[49]:


X


# In[50]:


X.shape


# checking VIF (Variance Inflation Factor)

# In[51]:


vif=pd.DataFrame()
vif["VIF Values"]= [variance_inflation_factor(X.values,i) for i in range(len(X.columns))]
vif["Features"]= X.columns
vif


# As we can see five columns are having VIF value more than 10 , so we are going to remove ER column as VIF in this column is too high

# In[52]:


X.drop("ER", axis=1 , inplace=True)


# In[53]:


# again checking vif value


# In[54]:


vif=pd.DataFrame()
vif["VIF Value"]=[variance_inflation_factor(X.values,i)for i in range(len(X.columns))]
vif["Independent variable"]=X.columns
vif


# As we can see two columns are having VIF value more than 10 , so we are going to remove RA column as VIF in this column is too high

# In[55]:


X.drop("RA", axis=1 , inplace=True)


# In[56]:


vif=pd.DataFrame()
vif["VIF Value"]=[variance_inflation_factor(X.values,i)for i in range(len(X.columns))]
vif["Independent variable"]=X.columns
vif


# Now the VIF value lies under 10

# # Modelling

# In[57]:


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
    


# In[58]:


# RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(X_train,Y_train)
pred_rf=rf.predict(X_test)
pred_train=rf.predict(X_train)
print("R2 Score  : ",r2_score(Y_test,pred_rf))
print("R2_score on training data :",r2_score(Y_train,pred_train)*100)
print("Mean Absolute Error : ", mean_absolute_error(Y_test,pred_rf))
print("Mean Squared Error : ",mean_squared_error(Y_test,pred_rf))
print("Root Mean Squared Error",np.sqrt(mean_squared_error(Y_test,pred_rf)))


# In[59]:


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


# In[60]:


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


# In[61]:


#KNeighborsRegressor Model
knn=KNN()
knn.fit(X_train,Y_train)
pred_knn=knn.predict(X_test)
pred_train=GBR.predict(X_train)
print("R2 Score  : ",r2_score(Y_test,pred_knn))
print("R2_score on training data :",r2_score(Y_train,pred_train)*100)
print("Mean Absolute Error : ", mean_absolute_error(Y_test,pred_knn))
print("Mean Squared Error : ",mean_squared_error(Y_test,pred_knn))
print("Root Mean Squared Error",np.sqrt(mean_squared_error(Y_test,pred_knn)))


# In[62]:


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


# In[63]:


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


# In[64]:


#DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(X_train,Y_train)
pred_dtr = dtr.predict(X_test)
pred_train= dtr.predict(X_train)
print('R2_score :',r2_score(Y_test,pred_dtr))
print('R2_score on training Data :', r2_score(Y_train,pred_train)*100)
print('Mean Absolute Error : ', mean_absolute_error(Y_test,pred_dtr))
print('Mean Squared error :-',mean_squared_error(Y_test,pred_dtr))
print("Root mean squared error :-",np.sqrt(mean_squared_error(Y_test,pred_dtr)))


# In[65]:


#SupportVectorRegressor
svr=SVR()
dtr.fit(X_train,Y_train)
pred_svr= dtr.predict(X_test)
pred_train= dtr.predict(X_train)
print('R2_score :',r2_score(Y_test,pred_svr))
print('R2_score on training Data :', r2_score(Y_train,pred_train)*100)
print('Mean Absolute Error : ', mean_absolute_error(Y_test,pred_svr))
print('Mean Squared error :-',mean_squared_error(Y_test,pred_svr))
print("Root mean squared error :-",np.sqrt(mean_squared_error(Y_test,pred_svr)))


# In[66]:


#ExtraTreeRegressor
rtr=ExtraTreesRegressor()
dtr.fit(X_train,Y_train)
pred_rtr= dtr.predict(X_test)
pred_train= dtr.predict(X_train)
print('R2_score :',r2_score(Y_test,pred_rtr))
print('R2_score on training Data :', r2_score(Y_train,pred_train)*100)
print('Mean Absolute Error : ', mean_absolute_error(Y_test,pred_rtr))
print('Mean Squared error :-',mean_squared_error(Y_test,pred_rtr))
print("Root mean squared error :-",np.sqrt(mean_squared_error(Y_test,pred_rtr)))


# In[67]:


# lets check Cross val score


# In[68]:


score=cross_val_score(lr,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_lr)-score.mean())*100)


# In[69]:


score=cross_val_score(rf,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_rf)-score.mean())*100)


# In[70]:


score=cross_val_score(knn,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_knn)-score.mean())*100)


# In[71]:


score=cross_val_score(GBR,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_GBR)-score.mean())*100)


# In[72]:


score=cross_val_score(lasso,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_lasso)-score.mean())*100)


# In[73]:


score=cross_val_score(rd,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_rd)-score.mean())*100)


# In[74]:


score=cross_val_score(dtr,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_dtr)-score.mean())*100)


# In[75]:


score=cross_val_score(svr,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_svr)-score.mean())*100)


# In[76]:


score=cross_val_score(rtr,X,Y)
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is ",(r2_score(Y_test,pred_rtr)-score.mean())*100)


# from the difference of both R2 and cross validation score we have consluded that Lasso is the best performing model

# In[79]:


param={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
gscv=GridSearchCV(Lasso(),param,scoring='neg_mean_squared_error',cv=5)
gscv.fit(X_train,Y_train)


# In[80]:


gscv.best_params_


# In[81]:


Model=Lasso(alpha=1)


# In[82]:


Model.fit(X_train,Y_train)
pred=Model.predict(X_test)
print('R2_score',r2_score(Y_test,pred))
print("Mean absolute Error :-",mean_absolute_error(Y_test,pred))
print("Mean Squared Error :-",mean_squared_error(Y_test,pred))
print("Root Mean Squared Error :-",np.sqrt(mean_squared_error(Y_test,pred)))


# In[83]:


import pickle
filename="Baseball Project"
pickle.dump(Model,open(filename,'wb'))


# In[84]:


loaded_model=pickle.load(open("Baseball Project","rb"))
result=loaded_model.score(X_test,Y_test)
print(result*100)


# In[85]:


final=pd.DataFrame([loaded_model.predict(X_test)[:],Y_test[:]],index=["Predicted","Original"])
final


# In[ ]:




