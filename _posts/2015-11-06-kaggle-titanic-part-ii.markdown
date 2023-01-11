---
layout: article
title: "Titanic Part II"
subtitle: "Machine Learning From Disaster"
tags: python kaggle
---

<!--more-->

```python
import csv
import numpy as np
import pandas as pd
import pylab as plt
import re
from patsy import dmatrices
%matplotlib inline  
train = pd.read_csv('train.csv', header = 0)
test = pd.read_csv('test.csv', header = 0)
```


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 90.5+ KB



```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 418 entries, 0 to 417
    Data columns (total 11 columns):
    PassengerId    418 non-null int64
    Pclass         418 non-null int64
    Name           418 non-null object
    Sex            418 non-null object
    Age            332 non-null float64
    SibSp          418 non-null int64
    Parch          418 non-null int64
    Ticket         418 non-null object
    Fare           417 non-null float64
    Cabin          91 non-null object
    Embarked       418 non-null object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 39.2+ KB


### Data Cleaning
In last post, we have done some exploratory analysis of the data and also built two simple models, which gave us decent prediction accuracy. In this post, our question is, Can we do better? 
We left off last time with Logistic Regression. And hopefully we just need to feed more predicators to the model for a better accuracy. However, before we can do that, we need to clean our data. We see that there are NAs in the Age, Fare, Cabin, and Embarked variables, so we will have to impute these values. For the Cabin variable, there are way too many NAs so imputing all those values would bring it an significant amount of noise. And the Cabin variable itself is hard to interpret anyways. So we will just impute the Age, Fare and Embarked variables.


```python
def clean_data(df):
    # drop predicators that we are not going to use
    df.drop(['Ticket','Cabin'],inplace=True,axis=1)
    
    # encode the sex class to 0, 1
    df['Sex'] = df.Sex.map( {'female': 0, 'male': 1} ).astype(int)

    # simply fill the missing Embarked value with "S" since there are only 2 NAs
    df['Embarked'] = df.Embarked.fillna('S')
    
    # there is a 0 value in Fare column, which is unreasonable. Will have to replace it with NA first.
    df.Fare = df.Fare.map(lambda x: np.nan if x==0 else x)
    # build a pivot table to impute NAs with averaged Fare for all Pclass
    fare_pivot_table = df.pivot_table("Fare", index='Pclass', aggfunc='mean', dropna=True)
    # use pivot table to impute missing fare values
    df['Fare'] = df[['Fare', 'Pclass']].apply(lambda x: fare_pivot_table[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis = 1)
    
    #build a pivot table to impute NAs with median Age according to Pclass and Sex
    age_pivot_table = df.pivot_table('Age', index=['Pclass','Sex'], aggfunc='median',dropna=True)
    df['Age'] = df[['Sex', 'Pclass', 'Age']].apply(lambda x: age_pivot_table[x.Pclass, x.Sex] if pd.isnull(x.Age) else x.Age, axis = 1)
    
    # define a age group categorical variable
    df['AgeGroup'] = 'adult'
    df.loc[ (df['Age']<=10) ,'AgeGroup'] = 'child'
    df.loc[ (df['Age']>=60) ,'AgeGroup'] = 'senior'
    
    # define a fare group categorical variable
    df['FareGroup'] = 'low'
    df.loc[ (df['Fare']<=20) & (df['Fare']>10) ,'FareGroup'] = 'mid'
    df.loc[ (df['Fare']<=30) & (df['Fare']>20) ,'FareGroup'] = 'mid-high'
    df.loc[ (df['Fare']>30) ,'FareGroup'] = 'high'
    
    # Here we are adding interaction terms between predicators. I have included a few reasonable interaction terms in my mind 
    df['Family_Size']=train['SibSp']+train['Parch']+1
    
    # Generating the Title predicator
    df['Title'] = df['Name'].apply(lambda x: getTitle(x))
    df['Title'] = df.apply(getAndReplaceTitle, axis=1)

    return df

def getTitle(name):
    # use regex to match the titles in passenger names
    # titles are the characters preceded by a comma and a space (", ") and succeeded by a period (".") in a passenger's name.
    # Here we use the positive lookbehind assertion to find the preceding (", ") and the positive lookahead assertion to match the succeeding (".").
    m = re.search('(?<=,\s)[a-zA-z\s]+(?=\.)', name)
    if m is None: 
        return np.nan
    else:
        return m.group(0)
    
def getAndReplaceTitle(df):
    # combine similar titles
    # the majority of the titles are pretty rare, which doesn't give much information about the passenger.
    title = df['Title']
    if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col','Sir']:
        return 'Mr'
    elif title in ['the Countess', 'Mme','Mrs','Lady','Dona']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms','Miss']:
        return 'Miss'
    elif title =='Dr':
            if df['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
    else:
        return title

```

Not only did we impute the missing values, we also added in lots of additional predicators that could be correlated to the survival rate, since the correlation between survival rate and predicators might not be fully captured by the variables we had. Note that introducing new predicators, especaiily new categorical predicators, would increases the dimensionality of the data, which could potentially impact our prediction accuracy due to curse of dimensionality. Therefore we will use feature selection technique in our models


```python
train = clean_data(train)
test = clean_data(test)
```

    /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/ipykernel/__main__.py:16: FutureWarning: scalar indexers for index type Int64Index should be integers and not floating point



```python
#define a function in order to simplify future predictions and output
def predictAndOutput(estimator, output_filename):
        estimator.fit(train_data.values, train_response.values.ravel())
        output = estimator.predict(test_data).astype(int)
        submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": output})
        submission.to_csv(output_filename, index=False)
        return output
```

## Logistic Regression


```python
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
formula = 'Survived ~ C(Pclass) + Sex  + Age + SibSp + Parch + Family_Size + Fare + C(Embarked)+ C(Title) + C(AgeGroup)'
train_response,train_data = dmatrices(formula, data=train, return_type="dataframe")
test['Survived'] = 1 #insert a dummy survived variable in order to create dmatrices
test_response,test_data = dmatrices(formula, data=test, return_type="dataframe")

LRModel = LogisticRegression()
# Exhaustive search over all parameter combinations for the estimator that produce the best CV error
param_grid = {'penalty':['l1','l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
GridSearch = GridSearchCV(LRModel, param_grid, cv = 10)
GridSearch.fit(train_data.values, train_response.values.ravel())
bestLRclf = GridSearch.best_estimator_
LRprediction = predictAndOutput(bestLRclf, "LROutput.csv")
```

We got **0.77990** this times, which is an great improvement over the 2 simple models we had in last post. Hooray, moving up the ladder! Let's take a look at the classifier that gave us the best results. 


```python
for i in range(1,len(train_data.columns.values)):
    print train_data.columns.values[i] , bestLRclf.coef_[0][i]
```

    C(Pclass)[T.2] -0.905477959537
    C(Pclass)[T.3] -1.97896459743
    C(Embarked)[T.Q] 0.0
    C(Embarked)[T.S] -0.299162084187
    C(Title)[T.Miss] -0.573336914082
    C(Title)[T.Mr] -2.58797777339
    C(Title)[T.Mrs] 0.0
    C(AgeGroup)[T.child] 0.284813224479
    C(AgeGroup)[T.senior] 0.0
    Sex -0.78528900026
    Age -0.025393089303
    SibSp -0.49647079287
    Parch -0.266397881617
    Family_Size 0.0
    Fare 0.00330362443408


We can tell from the coefficients of the predicators that some of the coefficients has been regularized to 0, which means that the predicator was not selected by the model. For the coefficients we can also observe the effect of the predicators. For example, passenger of Pclass 2 or 3would have a lower survival rate compared with passenger of Pclass 1 due to the negative coefficients. 

## Random Forest

Now we will move on to the more complicated Ensemble models, which usually offer the best out-of-box prediction accuracy. [Random Forest](https://en.wikipedia.org/wiki/Random_forest) is one of the most commonly-used Ensemble method that is operated by constructing a random multitude of decision trees at training time and outputting the class that is the mode of the classes.


```python
from sklearn.ensemble import RandomForestClassifier
RFModel = RandomForestClassifier(random_state=1, n_estimators=200)
param_grid = {'max_features':['sqrt',2, 5, 10, 15], 'min_samples_split':[4, 8, 10], 'min_samples_leaf':[2, 4, 6]}
GridSearch = GridSearchCV(RFModel, param_grid, cv = 10)
GridSearch.fit(train_data.values, train_response.values.ravel())
bestRFclf = GridSearch.best_estimator_
RFprediction = predictAndOutput(bestRFclf, "RFOutput.csv")
```

The most important parameters that we can tune for Random Forest is max_features, which is the number of features to consider when looking for the best split. Decreasing the max_features could improve randomness and reduce variance, while increasing the max_features would move the random forest more towards bagging. Min_samples_split and Min_samples_leaf are two stopping criteria of Random Forest that could contribute to overfitting. The GridSearchCV here is an exhaustive search over all parameter combinations for the estimator that produce the best 10-fold CV error.
The Random Forest Method achieved an accuracy of **0.77990** again.


```python
indices = np.argsort(bestRFclf.feature_importances_)[::-1]
plt.figure()
plt.title("Feature importances")
plt.barh(range(len(train_data.columns.values)), bestRFclf.feature_importances_[indices], color="r")
plt.yticks(range(len(train_data.columns.values)), train_data.columns.values[indices])
plt.show()
```


![png](	
https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post2/plot1.png)


This plot illustrates the feature importantces generated from Random Forest. It is interesting that the dummy variable of Title Mr. is the most important feature, while Title Mrs. and Title Miss. are among the least influential features. The Age, Fare, Pclass and Sex variables also rank high in terms of importances, which is expected.
## Blend Models
We have tried our hands at two models so far and they produced the exact same accuracy. So do both of the models make the same prediction on every test passengers? If not, we could somehow blend these two models together for a better result. 


```python
# Generating a list of test passenger indexes that have different predictions from the two models
different = list()
for i in range(len(RFprediction)):
    if RFprediction[i] != LRprediction[i]:
        different.append(i)
print different
print "Proportion of the predictions that the two models disagree on is " , len(different)/float(481)
```

    [1, 6, 19, 28, 32, 33, 34, 36, 37, 39, 41, 55, 63, 72, 86, 87, 88, 90, 94, 127, 132, 138, 144, 148, 158, 159, 169, 192, 199, 206, 249, 252, 268, 280, 291, 309, 313, 323, 344, 347, 367, 379, 382, 389, 390, 403, 409, 412, 417]
    Proportion of the predictions that the two models disagree on is  0.101871101871


It does turn out that the two models didn't make the same prediction on every passengers, as the output shows that the models diagree on about 10% of prediction. This is quite desirable since we are very confident on the rest 90% of the predictions. Now we just needed to find a way to blend the two models together.  


```python
LRProba = bestLRclf.predict_proba(test_data)
print LRProba[different[1:5]]
```

    [[ 0.33960866  0.66039134]
     [ 0.29814563  0.70185437]
     [ 0.6590034   0.3409966 ]
     [ 0.53082693  0.46917307]]


For a ensemble method like Random Forest, the interpretbility is very low. However, the Logistic Regression method is actually very intrepretible, as we can compute the probabiliities of the binary responses for each predictions. Here we can see that for the 2nd prediction that the models differed on, the probabilities for 0 and 1 are 0.298 and 0.702, which means that the logistic regression was quite confident that this passengers was gonna survive, whereas for the 5th prediction that the models differed on, the probabilities for 0 and 1 are 0.530 and 0.469, the logistic regression was a lot less confident. Using the probabilities, we could derive an simple method to predict when the two models disagree. 


```python
from copy import deepcopy
CombinePrediction = deepcopy(RFprediction)
for i in range(len(different)):
    # If the probabilties of survival is greater than 0.7, predict survived
    if LRProba[different[i]][0] > 0.7:
        CombinePrediction[different[i]] = 0
    # If the probabilties of survival is less than 0.3, predict perished   
    elif LRProba[different[i]][0] < 0.3:
        CombinePrediction[different[i]] = 1
    else:
        continue
submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": CombinePrediction})
submission.to_csv("CombineOutput.csv", index=False)
```

This gives us **0.79904**, the best we have had so far and top 25% on the public leaderboard. 
## Areas of Improvement
Of course there is still a lot of room for improvement. 
- Better tunning of parameters on Logistic Regression and Random Forest. Some people could use one single method to achieve 0.81 or above. Maybe include  [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) in the CV process.
- Do a few models and combine them using majority vote. Maybe SVM, or Gradient Boosting. 
- Better imputation on missing values
- Make use of familiy lastnames as a predicator.
