# import all libraries needed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import itertools

class titanic_model():
    
    def clean_data(self, dataset):
        # Drop the str type columns
        dataset.drop(columns={'Cabin','Name','Ticket'},inplace=True)
        dataset.dropna(subset=['Embarked'],inplace=True)  
        # Replace the NaN value in int column with it's mean
        age_mean = int(dataset['Age'].mean())
        fare_mean = dataset['Fare'].mean()
        dataset['Age'].fillna(value=age_mean,inplace=True)
        dataset['Fare'].fillna(value=fare_mean,inplace=True)
        # Check if the missing value still exist :
        for k, v in dataset.iteritems():
            if v.empty is True:
                print('YOUR DATA STILL NOT CLEAN, DUDE!!!')
            else:
                print('YES, YOUR DATA WAS ALREADY CLEAN ON COLUMN', k)
        return self
    
    def correlation(self,dataset):
        # Visualize Data Correlation
        f = plt.figure(figsize=(19, 15))
        plt.matshow(dataset.corr(),fignum=f.number)
        plt.xticks(range(dataset.shape[1]),dataset.columns,fontsize=14,rotation=75)
        plt.yticks(range(dataset.shape[1]),dataset.columns,fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        return self

    def survived_gender(self,dataset):
        #Survived and not survived Passenger based on gender
        total_survived = sum(dataset.Survived)
        men = dataset.loc[dataset.Sex == 'male']['Survived']
        men_survived = sum(men) 
        women = dataset.loc[dataset.Sex == 'female']['Survived']
        women_survived = sum(women)
        labels = ['Women Survived','Men Survived']
        values = [women_survived,men_survived]
        not_survived = int(len(dataset)) - total_survived
        labels1 = ['Women Deaths','Men Deaths']
        values1 = [int(len(women)) - women_survived, int(len(men)) - men_survived]
        # visualize the data, make pie chart
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
        fig.add_trace(go.Pie(labels=labels, values=values, name="Survived"), 1, 1)
        fig.add_trace(go.Pie(labels=labels1, values=values1, name="Not Survived"), 1, 2) 
        # Use `hole` to create a donut-like pie chart
        fig.update_traces(hole=.65, hoverinfo="label+percent+name")
        fig.update_layout(
            title_text="Passenger Survival Percentage based on Wealth",
            # Add annotations in the center of the donut pies.
            annotations=[dict(text='Survived :' + str(total_survived), x=0.12, y=0.5, font_size=12, showarrow=False),
                         dict(text='Not Survived :' + str(not_survived), x=0.91, y=0.5, font_size=12, showarrow=False)])
        fig.show()
        return self

    def survived_class(self,dataset):
        Pclass_1 = 0
        Pclass_2 = 0
        Pclass_3 = 0
        Pclass = dataset.loc[dataset.Survived == 1]['Pclass']
        for key, value in Pclass.iteritems():
            if value == 1:
                Pclass_1 += 1
            if value == 2:
                Pclass_2 += 1
            if value == 3:
                Pclass_3 += 1
        Plabels = ['Class 1 Survived','Class 2 Survived','Class 3 Survived']
        Pvalues = [Pclass_1, Pclass_2, Pclass_3]
        # Passenger who not survived based on Class
        Nclass = dataset.loc[dataset.Survived == 0]['Pclass']
        Nclass_1 = 0
        Nclass_2 = 0 
        Nclass_3 = 0
        for key, value in Nclass.iteritems():
            if value == 1:
                Nclass_1 += 1
            if value == 2:
                Nclass_2 += 1
            if value == 3:
                Nclass_3 += 1
        Nlabels = ['Class 1 not Survived','Class 2 not Survived','Class 3 not Survived']
        Nvalues = [Nclass_1, Nclass_2, Nclass_3]
        # visualize the data, make pie chart
        total_survived = sum(dataset.Survived)
        not_survived = int(len(dataset)) - total_survived
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
        fig.add_trace(go.Pie(labels=Plabels, values=Pvalues, name="Survived"),1, 1)
        fig.add_trace(go.Pie(labels=Nlabels, values=Nvalues, name="Not Survived"),1, 2)
        # Use `hole` to create a donut-like pie chart
        fig.update_traces(hole=.65, hoverinfo="label+percent+name")
        fig.update_layout(
            title_text="Passenger Survival Percentage based on Wealth",
        # Add annotations in the center of the donut pies.
            annotations=[dict(text='Survived :' + str(total_survived), x=0.12, y=0.5, font_size=12, showarrow=False),
            dict(text='Not Survived :' + str(not_survived), x=0.91, y=0.5, font_size=12, showarrow=False)])
        fig.show()
        return self

    def modeling(self,dataset):
        # Labeling the string values like sex and embarked
        LE = LabelEncoder()
        dataset.Sex = LE.fit_transform(dataset.Sex)
        dataset.Embarked = LE.fit_transform(dataset.Embarked)
        # Split the data into x and y, and use standard scaler 
        x = dataset[['PassengerId','Pclass','Fare','Sex','Age','SibSp','Parch','Embarked']]
        y = dataset['Survived']
        x = StandardScaler().fit(x).transform(x)
        #split into train and test data, and fit the model using Logistic Regression
        x_train, x_test, y_train, y_test = train_test_split( x, y,test_size = 0.319816373,random_state = 150)
        LR = LogisticRegression(C=0.05, solver='liblinear').fit(x_train, y_train)
        y_pred = LR.predict(x_test) 
        print(classification_report(y_test, y_pred))
        print('Your model have accuracy of : ' + str(100*accuracy_score(y_test, y_pred)) + ' %')
        # Check the log loss of our prediction model
        y_pred_prob = LR.predict_proba(x_test)
        print('Your model have Log Loss of :', str(100*(log_loss(y_test, y_pred_prob))) +' %')
        #Confusion Matrix
        classes=[0,1]
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        print('Confusin Matrix of your Model is :')
        print(cm)
        return self

