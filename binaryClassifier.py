# In[1]: Header and load data


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train_df=pd.read_csv("data/train.csv")
# train_df.head()
test_df=pd.read_csv("data/test.csv")
train_df.head()


# In[2]: Find and show missing data


def missingdata(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms= ms[ms["Percent"] > 0]
    f,ax =plt.subplots(figsize=(8,6))
    plt.xticks(rotation='90')
    fig=sns.barplot(ms.index, ms["Percent"],color="green",alpha=0.8)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    return ms
missingdata(test_df)


# In[3]: Data Cleaning


train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)
train_df['Age'].fillna(train_df['Age'].median(), inplace = True)
drop_column = ['Cabin']
train_df.drop(drop_column, axis=1, inplace = True)
print('check the nan value in train data')
print(train_df.isnull().sum())

test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace = True)
test_df['Age'].fillna(test_df['Age'].median(), inplace = True)
test_df['Fare'].fillna(test_df['Fare'].mode()[0], inplace = True)
drop_column = ['Cabin']
test_df.drop(drop_column, axis=1, inplace = True)
print('check the nan value in train data')
print(test_df.isnull().sum())


# In[7]: Feauture Engineering


dataset = train_df
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

import re
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don','Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
dataset['Age_bin'] = pd.cut(dataset['Age'], bins=[0,14,20,40,120], labels=['Children','Teenage','Adult','Elder'])
dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','median_fare', 'Average_fare','high_fare'])
                                                                               
traindf=train_df
drop_column = ['Age','Fare','Name','Ticket']
train_df.drop(drop_column, axis=1, inplace = True)
drop_column = ['PassengerId']
traindf.drop(drop_column, axis=1, inplace = True)
traindf = pd.get_dummies(traindf, columns = ["Sex","Title","Age_bin","Embarked","Fare_bin"],
                             prefix=["Sex","Title","Age_type","Em_type","Fare_type"])

sns.heatmap(traindf.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()

dataset = test_df
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

import re
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don','Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
dataset['Age_bin'] = pd.cut(dataset['Age'], bins=[0,14,20,40,120], labels=['Children','Teenage','Adult','Elder'])
dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','median_fare', 'Average_fare','high_fare'])
                                                                               
testdf=test_df
drop_column = ['Age','Fare','Name','Ticket']
test_df.drop(drop_column, axis=1, inplace = True)
drop_column = ['PassengerId']
index_df = test_df['PassengerId'].values
testdf.drop(drop_column, axis=1, inplace = True)
testdf = pd.get_dummies(testdf, columns = ["Sex","Title","Age_bin","Embarked","Fare_bin"],
                             prefix=["Sex","Title","Age_type","Em_type","Fare_type"])

sns.heatmap(testdf.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(20,12)
plt.show()

# In[14]: Making train and test data ready

from sklearn.model_selection import train_test_split #for split the data
from sklearn.metrics import accuracy_score  #for accuracy_score
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.metrics import confusion_matrix #for confusion matrix
all_features = traindf.drop("Survived",axis=1)
Targeted_feature = traindf["Survived"]
X_train,X_test,y_train,y_test = train_test_split(all_features,Targeted_feature)
X_train.shape,X_test.shape,y_train.shape,y_test.shape,testdf.shape

# In[]: SVM

from sklearn import svm
clf = svm.SVC(gamma='scale').fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the SVM Classifier is', round(accuracy_score(prediction_rm,y_test)*100,2))
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
result_rm=cross_val_score(clf,all_features,Targeted_feature,cv=10,scoring='accuracy')
print('The cross validated score for SVM Classifier is:',round(result_rm.mean()*100,2))
y_pred = cross_val_predict(clf,all_features,Targeted_feature,cv=10)
sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="summer")
plt.title('Confusion_matrix', y=1.05, size=15)

prediction_on_test_rm=clf.predict(testdf)
prediction_on_test_rm
type(prediction_on_test_rm)
index_df
type(index_df)
pd.DataFrame(data=np.hstack((index_df[:,None], prediction_on_test_rm[:,None])),
            columns=['PassengerId','Survived']).to_csv('svm.csv', index=False)

# In[15]: Logistic Regression

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
prediction_rm=clf.predict(X_test)
print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the Logistic Regression Classifier is', round(accuracy_score(prediction_rm,y_test)*100,2))
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
result_rm=cross_val_score(clf,all_features,Targeted_feature,cv=10,scoring='accuracy')
print('The cross validated score for Logistic Regression Classifier is:',round(result_rm.mean()*100,2))
y_pred = cross_val_predict(clf,all_features,Targeted_feature,cv=10)
sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="summer")
plt.title('Confusion_matrix', y=1.05, size=15)



# In[16]: Random Forest Learning


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(criterion='gini', n_estimators=700,
                             min_samples_split=10,min_samples_leaf=1,
                             max_features='auto',oob_score=True,
                             random_state=1,n_jobs=-1)
model.fit(X_train,y_train)
prediction_rm=model.predict(X_test)
print('--------------The Accuracy of the model----------------------------')
print('The accuracy of the Random Forest Classifier is', round(accuracy_score(prediction_rm,y_test)*100,2))
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
result_rm=cross_val_score(model,all_features,Targeted_feature,cv=10,scoring='accuracy')
print('The cross validated score for Random Forest Classifier is:',round(result_rm.mean()*100,2))
y_pred = cross_val_predict(model,all_features,Targeted_feature,cv=10)
sns.heatmap(confusion_matrix(Targeted_feature,y_pred),annot=True,fmt='3.0f',cmap="summer")
plt.title('Confusion_matrix', y=1.05, size=15)


# In[17]: Tuning


model = RandomForestClassifier(criterion='gini', n_estimators=700,
                             min_samples_split=10,min_samples_leaf=1,
                             max_features='auto',oob_score=True,
                             random_state=1,n_jobs=-1)


from sklearn.model_selection import GridSearchCV# Random Forest Classifier Parameters tunning 
model = RandomForestClassifier()
n_estim=range(100,1000,100)## Search grid for optimal parameters
param_grid = {"n_estimators" :n_estim}
model_rf = GridSearchCV(model,param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)
model_rf.fit(X_train,y_train)# Best score
print(model_rf.best_score_)#best estimator
model_rf.best_estimator_


# In[20]:
def plotData2D(X, filename=None):
    fig = plt.figure()
    axs = fig.add_subplot(111)

    axs.plot(X[0, :], X[1, :], 'ro', label='data')
    if filename == None:
        plt.show()
    plt.close()


def plotClusters(X, k, clusters, filename=None):
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    # plot the data
    colors = ["red", "green", "blue", "yellow"]
    for i in range(np.size(X, 1)):
        plt.scatter(X[0][i], X[1][i], color=colors[clusters[i]])

    # either show figure on screen or write it to disk
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()


def linearClassifier(X, y):  # TODO: lift data
    W = np.linalg.inv(np.dot(X, X.T))
    W = np.dot(np.dot(W, X), y)
    return np.dot(X.T, W), np.sum(np.power(np.dot(X.T, W) - y, 2))/np.size(X, 1)
