#%%
#Import libraries and data
from numpy.core.defchararray import index
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv(r'C:\Users\USER\Documents\Belajar Python 3.8\00 - Template\titanic\train.csv')
test_data = pd.read_csv(r'C:\Users\USER\Documents\Belajar Python 3.8\00 - Template\titanic\test.csv')
#Check Missing Values
for key, value in train_data.iteritems():
    print(key, ':', str(value.isnull().sum()))
for col in train_data.columns:
    print(col, str(round(100*train_data[col].isnull().sum()/len(train_data))) + '%')

#%%
#Check Missing Values
for key, value in train_data.iteritems():
    print(key, ':', str(value.isnull().sum()))
for col in train_data.columns:
    print(col, str(round(100*train_data[col].isnull().sum()/len(train_data))) + '%')

#clean the train data:
train_data.drop(columns='Cabin',inplace=True)
train_data.dropna(axis=0,how='any',subset=['Embarked'],inplace=True)
mean_age = int(train_data['Age'].mean())

for key, value in train_data['Age'].iteritems():
    if value > 0:
        pass
    else:
        train_data['Age'].replace(value,mean_age,inplace=True)

#Clean the test data
age_mean = test_data['Age'].mean()
fare_mean = test_data['Fare'].mean()
test_data['Fare'].fillna(value=fare_mean,inplace=True)
test_data.drop(columns='Cabin',inplace=True)
for key, value in test_data['Age'].iteritems():
    if value > 0:
        pass
    else:
        test_data['Age'].replace(value,age_mean,inplace=True)

#Analyze Survived Passangger based on Gender
total_survived = sum(train_data.Survived)
men = train_data.loc[train_data.Sex == 'male']['Survived']
men_survived = sum(men)
women = train_data.loc[train_data.Sex == 'female']['Survived']
women_survived = sum(women)

import plotly.express as px
labels = ['Women Survived','Men Survived']
values = [women_survived,men_survived]
fig = px.pie(
     values=values,
     names=labels, 
     title='Total Survived from Shipcrash:' + str(total_survived), 
     hole = 0.7,
     template='plotly_dark',
)
fig.show()
#Not Survived Passanger based on Gender
not_survived = int(len(train_data)) - total_survived
labels = ['Women Deaths','Men Deaths']
values = [int(len(women)) - women_survived, int(len(men)) - men_survived]
fig = px.pie(
     values=values,
     names=labels, 
     title='Total Not Survived from Shipcrash:' + str(not_survived), 
     hole = 0.7,
     template='plotly_dark',
)
fig.show()

#Analyze survived passanger based on their wealth
Pclass = train_data.loc[train_data.Survived == 1]['Pclass']
class_1 = 0
class_2 = 0
class_3 = 0
for key, value in Pclass.iteritems():
    if value == 1:
        class_1 = class_1 + 1
    if value == 2:
        class_2 = class_2 + 1
    if value == 3:
        class_3 = class_3 + 1
labels = ['Class 1 Survived','Class 2 Survived','Class 3 Survived']
values = [class_1, class_2, class_3]
fig = px.pie(
     values=values,
     names=labels, 
     title='Total Survived from Shipcrash:' + str(total_survived), 
     hole = 0.7,
     template='plotly_dark',
)
fig.show()

Nclass = train_data.loc[train_data.Survived == 0]['Pclass']
class_1 = 0
class_2 = 0
class_3 = 0
for key, value in Nclass.iteritems():
    if value == 1:
        class_1 = class_1 + 1
    if value == 2:
        class_2 = class_2 + 1
    if value == 3:
        class_3 = class_3 + 1
labels = ['Class 1 not Survived','Class 2 not Survived','Class 3 not Survived']
values = [class_1, class_2, class_3]
fig = px.pie( 
     values=values,
     names=labels, 
     title='Total Not Survived from Shipcrash:' + str(not_survived), 
     hole = 0.7,
     template='plotly_dark',
)
fig.show()

train = train_data[['PassengerId','Pclass','Fare','Sex','Age','SibSp','Parch','Survived']]
test = test_data[['PassengerId','Pclass','Fare','Sex','Age','SibSp','Parch']]
y_test = pd.read_csv(r'C:\Users\USER\Documents\Belajar Python 3.8\00 - Template\titanic\gender_submission.csv')
TEST = pd.merge(test,y_test,on='PassengerId')
new_data = pd.concat([train,TEST])
new_data.reset_index(drop=True,inplace=True)
LE = LabelEncoder()
new_data.Sex = LE.fit_transform(new_data.Sex)
new_data.corr(method='pearson')

f = plt.figure(figsize=(19, 15))
plt.matshow(new_data.corr(),fignum=f.number)
plt.xticks(range(new_data.shape[1]),new_data.columns,fontsize=14,rotation=75)
plt.yticks(range(new_data.shape[1]),new_data.columns,fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)

import seaborn as sns
sns.pairplot(
    data=new_data,
    hue='Survived',
    vars=['Pclass','Fare','Sex','Age','SibSp','Parch'],
    kind='scatter')


X = np.array(new_data[['Pclass','Fare','Sex','Age','SibSp','Parch']])
Y = np.array(new_data['Survived'])
X = StandardScaler().fit(X).transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.319816373,random_state=150)

LR = LogisticRegression(C=0.05, solver='liblinear').fit(X_train, Y_train)
Y_pred = LR.predict(X_test) 
print(classification_report(Y_test, Y_pred))
print('Your model have accuracy of : ' + str(100*accuracy_score(Y_test, Y_pred)) + ' %')

#Confusion Matrix
classes=[0,1]
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
confusion_matrix(Y_test,Y_pred,labels=classes)
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

cnf_matrix = confusion_matrix(Y_test, Y_pred, labels=classes)
plt.ylabel('True Layer Label')
plt.xlabel('Predicted Layer Label')
plot_confusion_matrix(cnf_matrix,classes=classes,normalize= False, title='Confusion matrix')

#Log Loss

Y_pred_prob = LR.predict_proba(X_test)
print('Your model have Log Loss of :', str(100*(log_loss(Y_test, Y_pred_prob))) +' %')


# %%
