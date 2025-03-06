#import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

#read the dataset
dataMoodsData = pd.read_csv('Apple Quality.csv')

#print the first 5 rows
print(dataMoodsData.shape)
print(dataMoodsData.head())

print(dataMoodsData.describe())
print(dataMoodsData.describe().T)

types = dataMoodsData.dtypes
print(types)

#Create a histogram of all the variables
dataMoodsData.hist(figsize = (20,20))
plt.show()

#Create a pairplot using three distinct attributes
X = dataMoodsData[['Ripeness','Sweetness','Size','Quality']] 
from pandas.plotting import scatter_matrix
sns.pairplot(X, hue = 'Quality', diag_kind = "kde")
plt.show()

#Create a scatter matrix using three distinct attributes
Z = dataMoodsData[['Ripeness','Sweetness','Size']] 
y = dataMoodsData[['Quality']]
from pandas.plotting import scatter_matrix
scatter_matrix(Z,figsize=(10, 10))
plt.show()


#Assign the attributes to the X and Y values
X = dataMoodsData[['Ripeness','Sweetness','Size']]
y = dataMoodsData[['Quality']]

X.isna().sum()

#Create the train and test values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

#Create the classifier and assign the metric
classifier = KNeighborsClassifier(n_neighbors=100, metric = "manhattan")
classifier.fit(X_train, np.ravel(y_train,order='C'))
y_pred = classifier.predict(X_test)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#Find accuracy 
accuracy = accuracy_score(y_test,y_pred)*100
print(accuracy)

#Assigning 150 as hyperparameter
k_range=range(1,150)
scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric = "manhattan")
    knn.fit(X_train,np.ravel(y_train,order='C'))
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))

print(scores)

knn_cv = KNeighborsClassifier(n_neighbors = 15, metric = "manhattan")
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

cv_scores = cross_val_score(knn_cv,X,np.ravel(y,order='C'),cv=10)
print(cv_scores)
print(np.mean(cv_scores))

#Plot the accuracy on a linegraph
plt.plot(k_range,scores)
plt.xlabel('Value of K')
plt.ylabel('Testing Accuracy')
plt.show()




sns.set(style="whitegrid")

dataMoodsData=pd.read_csv('Apple Quality.csv')

#Assign the dataset to the X and Y values
X=pd.DataFrame(dataMoodsData)
X=X.iloc[:,0:8]
print(X)

y=dataMoodsData['Quality']
print(y)

#Create Train and Test values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=12)

#Create the Random Forest Regressor Method
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train,y_train)
rf.feature_importances_
print(rf.feature_importances_)

sorted_idx=rf.feature_importances_.argsort()

#Plot the Random Forest Regressor
plt.barh(X_train.columns[sorted_idx],rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")

print(sorted_idx)

feature_scores = pd.Series(rf.feature_importances_,index=X_train.columns).sort_values(ascending=False)

f, ax = plt.subplots(figsize=(30, 24))
ax=sns.barplot(x=feature_scores, y=X_train.columns)
ax.set_title("Visualize feature scores of the features")
ax.set_yticklabels(feature_scores.index)
ax.set_xlabel("Feature importance score")
ax.set_ylabel("Features")
plt.show()



#Optimize the Data
dataMoodsData_Optimize = pd.read_csv('Apple Quality.csv')
dataMoodsData_Optimize.columns=dataMoodsData_Optimize.columns.to_series().apply(lambda x: x.strip())
X = dataMoodsData_Optimize[['A_id','Size','Weight','Sweetness','Crunchiness','Juiciness','Ripeness','Acidity']]
y = dataMoodsData_Optimize[['Quality']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

scores=[]
k_range = range(1,40)

#Cross Validate the Data
for k in k_range:
    knn_cv = KNeighborsClassifier(n_neighbors=k, metric = "manhattan")
    cv_scores = cross_val_score(knn_cv,X,np.ravel(y,order='C'),cv=10)
    print(k)
    print(cv_scores)
    print(np.mean(cv_scores))

#Print the Accuracy scores
knn = KNeighborsClassifier(n_neighbors=15, metric = "manhattan")
knn.fit(X_train,np.ravel(y_train,order='C'))
y_pred = knn.predict(X_test)
accuracy_scores = metrics.accuracy_score(y_test,y_pred)
print(accuracy_scores)

test_scores = []
train_scores = []

for i in range(1,15):
    knn = KNeighborsClassifier(i, metric = "manhattan")
    knn.fit(X_train,y_train)

    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))

max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

#Print max train and test scores
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))

max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x:x+1,test_scores_ind))))

#Print lineplot of Training and Test score
plt.figure(figsize=(12,5))
p = sns.lineplot(x = range(1,15),y = train_scores,marker='*',label='Train Score')
p = sns.lineplot(x = range(1,15),y = test_scores,marker='o',label='Test Score')
plt.show()

#Find Error
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors = i, metric = "manhattan")
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    pred_i = pred_i.reshape(1000,1)
    error.append(np.mean(pred_i != y_test))
    
#Create the Error Graph
plt.figure(figsize = (12,6))
plt.plot(range(1,40), error, color='red', linestyle = 'dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
