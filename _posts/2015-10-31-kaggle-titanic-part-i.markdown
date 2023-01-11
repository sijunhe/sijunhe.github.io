---
layout: article
title: "Titanic Part I"
subtitle: "Machine Learning From Disaster"
tags: python kaggle
---

<!--more-->

## Problem Statement 
> The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.  
> One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.  
> In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
>> [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)



``` python
import csv
import numpy as np
import pandas as pd
import pylab as p
%matplotlib inline  
train = pd.read_csv('train.csv', header = 0)
test = pd.read_csv('test.csv', header = 0)
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


From the info output, we could see that there were null values in the Age, Cabin and Embarked features. The majority of the values in Cabin feature were null and I felt the values were hard to interpret anyways so I decided to not include it in the model. There were also a few missing values in the Age and Embarked features, which would be imputed later.

```python
train.head()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>

```python
train[train['Age'].isnull()].head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>1</td>
      <td>2</td>
      <td>Williams, Mr. Charles Eugene</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>244373</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>1</td>
      <td>3</td>
      <td>Masselmani, Mrs. Fatima</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2649</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>0</td>
      <td>3</td>
      <td>Emir, Mr. Farred Chehab</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2631</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>1</td>
      <td>3</td>
      <td>O'Dwyer, Miss. Ellen "Nellie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330959</td>
      <td>7.8792</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>

```python
train[train['Embarked'].isnull()]
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61</th>
      <td>62</td>
      <td>1</td>
      <td>1</td>
      <td>Icard, Miss. Amelie</td>
      <td>female</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>113572</td>
      <td>80</td>
      <td>B28</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>829</th>
      <td>830</td>
      <td>1</td>
      <td>1</td>
      <td>Stone, Mrs. George Nelson (Martha Evelyn)</td>
      <td>female</td>
      <td>62</td>
      <td>0</td>
      <td>0</td>
      <td>113572</td>
      <td>80</td>
      <td>B28</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



So, let's dive deeper into the data and see if we could understand which features were more important for our model. The competition summary hinted that some groups of people were more likely to survive than others, such as women, children, and the upper-class. The obvious predicators in the data for women and children is sex and age. The Pclass predicator should be a direct representation of a passenger's socio-economic status. The Fare predicator might also indicates if a passenger was upper-class.

```python
fig, axes = p.subplots(nrows=2, ncols=2, figsize=(15,10))
#compare male and female survival rate
male_female1 = pd.DataFrame({'Male':train[train['Sex'] == 'male']['Survived'].value_counts(),
                             'Female':train[train['Sex'] == 'female']['Survived'].value_counts()}, columns = ['Male','Female'])
male_female1.plot(kind = 'bar', title='Male and Female Survival Comparison 1', alpha = 0.3, stacked = True, ax=axes[0,0])

male_female2 = pd.DataFrame({'Survived':train[train['Survived'] == 1]['Sex'].value_counts(),
                             'Died':train[train['Survived'] == 0]['Sex'].value_counts()}, columns = ['Survived','Died'])
male_female2.plot(kind = 'bar', title='Male and Female Survival Comparison 2', alpha = 0.3, stacked = True, ax=axes[0,1])

#compare male and female survival rate across different age groups
male_survived_age = pd.DataFrame({'Survived':train[(train['Sex'] == 'male') & (train['Survived'] == 1)]['Age'],
                                  'Died':train[(train['Sex'] == 'male') & (train['Survived'] == 0)]['Age']}, columns = ['Survived','Died'])
male_survived_age.plot(kind = 'hist', title='Male Survival Age Histogram', alpha = 0.3, stacked = True, ax=axes[1,0])

female_survived_age = pd.DataFrame({'Survived':train[(train['Sex'] == 'female') & (train['Survived'] == 1)]['Age'],
                                    'Died':train[(train['Sex'] == 'female') & (train['Survived'] == 0)]['Age']}, columns = ['Survived','Died'])
female_survived_age.plot(kind = 'hist', title='Female Survival Age Histogram', alpha = 0.3, stacked = True, ax=axes[1,1])
```

![png](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post1/plot1.png)


The first two plots showed that females had a significantly higher chance of survival compared with males, thanks to the "women and children first" maxim. Kudos to the gallant gentlemen who sacrificed themselves for the lives of the women and children. As we could see from the male survival histogram, male had a uniformly low survival rate across all age groups. It was noteworthy that the age group with the highest male survival rate was 0 - 10 years old, aka children, and elderly females almost all survived.


```python
fig, axes = p.subplots(nrows=1, ncols=2, figsize=(15,5))
#compare survival rate of different socio-economic class
survival_pclass = pd.DataFrame({'Survived':train[train['Survived'] == 1]['Pclass'].value_counts(),
                    'Died':train[train['Survived'] == 0]['Pclass'].value_counts()}, columns = ['Survived','Died'])
survival_pclass.plot(kind = 'bar', title='Pclass Survival Comparison', alpha = 0.3, stacked = True, ax=axes[0])

#compare survival rate of different socio-economic class and sex
class_sex = pd.DataFrame({'High-Mid Class Male':train[(train['Sex'] == 'male') & (train['Pclass'] != 1)]['Survived'].value_counts(),
                          'High-Mid Class Female':train[(train['Sex'] == 'female') & (train['Pclass'] != 3)]['Survived'].value_counts(),
                          'Low Class Male':train[(train['Sex'] == 'male') & (train['Pclass'] == 3)]['Survived'].value_counts(),
                         'Low Class Female':train[(train['Sex'] == 'female') & (train['Pclass'] == 3)]['Survived'].value_counts()},
                         columns = ['High-Mid Class Male','High-Mid Class Female','Low Class Male','Low Class Female'])
class_sex.plot(kind = 'bar', title='Pclass Sex Survival Histogram', alpha = 0.3, ax=axes[1])
```

![png](https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post1/plot2.png)


Here I generated two plots to show that if the survival rate of passengers are related to their socio-economic class. From the plot on the left, it was evident that passengers with a Pclass of 3, which means lower-class, had a much lower survival rate compared with passengers with a Pclass of 1 or 2. The plot on the right shows how the a combination of Pclass and Sex variables could influence the survival rate of a passenger. It is notable that the different socio-economic status only seemed to significantly affect the survival rate of females, not males. The odds of a upper-class lady surviving the Titanic shipwreck is surprising high.


```python
fig, axes = p.subplots(nrows=1, ncols=2, figsize=(15,5))
embarked_survival1 = pd.DataFrame(train[train['Survived'] == 1]['Embarked'].value_counts()/train['Embarked'].value_counts())
embarked_survival1.plot(linestyle='-', marker='o', ax=axes[0])
embarked_survival2 = pd.DataFrame({'Survived':train[train['Survived'] == 1]['Embarked'].value_counts(),
                                  'Died':train[train['Survived'] == 0]['Embarked'].value_counts()}, columns = ['Survived','Died'])
embarked_survival2[['Survived','Died']].plot(kind = 'bar', title='Male and Female Survival Comparison 1', alpha = 0.3, stacked = True, ax=axes[1])
```

![png](	
https://s3-us-west-1.amazonaws.com/sijunhe-blog/plots/post1/plot3.png)


I also generated two plots to check if different Port of Embarkation would make a difference in survival rate. It is surprising that it actually did have an impact. As passengers embarked from Cherbourg have a significant higher survival chance compared with passengers embarked from Queenstown or Southampton.

## Basic Models
With the exploratory visualization from the last point, we have some idea what out data looks like. We know that the survival rate is in some way related to Pclass, Sex, Age, Fare and Embarked, but we haven't got any chance to explore the rest of the predicators. So, let's start building models and making predictions.

### Purely Gender-Based Model
From last post, we know that the most influential predicator in the data set is Sex, as survival rate differs dramatically between male and female passengers. Given the low survival rate of male and high survival rate of female, the easiest possible model is to simply predict 0 for male and 1 for female.


```python
#simply creating a Survival variable, mappped from the Sex variable
test['Survived'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
```

The output of the head() function tells us our change to the data frame is effective and we have already made our first prediction! Next, we just need to export the data to a csv file and submit to Kaggle.


```python
prediction_file = open("Gender_Model.csv", "wb")
out = csv.writer(prediction_file)
out.writerow(["PassengerId", "Survived"])
for i in range(0, len(test.PassengerId)):
    out.writerow([test.PassengerId[i], test.Survived[i]])
prediction_file.close()
```

The results of the gender-based prediction was **0.76555**, which is surprisingly good for a single line of prediction code! Of course we are not going to call it a day. Let's see if we could incorporate Pclass or Age in the model.  
### Gender-Class Model
While we could do a pivot table, predicting the majority of each gender in different Pclass, I'd say it is time to bring in some models. Let's try our hands at [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression) first, since it is one of the most interpretable classification model in my opinion. We'll start with an extremely simple case of just 2 predicators: Pclass and Sex  


```python
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
#setting formula to predict Survived with Pclass and Sex
formula1 = 'Survived ~ C(Pclass) + C(Sex)'
train_response,train_data = dmatrices(formula1, data=train, return_type="dataframe")
test_response,test_data = dmatrices(formula1, data=test, return_type="dataframe")
LRmodel = LogisticRegression()
LRmodel.fit(train_data.values, train_response.values.ravel())
output = LRmodel.predict(test_data).astype(int)
```

It seems like we got the exact same result **0.76555** again. Why is that? Shouldn't an adding an predicator change some of our predictions? Here is why.


```python
for i in range(0,2):
    for j in range(0,2):
        for k in range(0,2):
            if i == 1:
                Pclass = 2
            elif j == 1:
                Pclass = 3
            else: 
                Pclass = 1
            if i == j ==1:
                break    
            if k ==1:
                sex = "male"
            else:
                sex = "female"
            print "Given Pclass = %d and Sex = %s, predict %d for survival. " % (Pclass,sex, LRmodel.predict([[1,i,j,k]])) 
```

    Given Pclass = 1 and Sex = female, predict 1 for survival. 
    Given Pclass = 1 and Sex = male, predict 0 for survival. 
    Given Pclass = 3 and Sex = female, predict 1 for survival. 
    Given Pclass = 3 and Sex = male, predict 0 for survival. 
    Given Pclass = 2 and Sex = female, predict 1 for survival. 
    Given Pclass = 2 and Sex = male, predict 0 for survival. 



```python
print train.pivot_table('Survived', index=['Pclass','Sex'], aggfunc='mean')
```

    Pclass  Sex   
    1       female    0.968085
            male      0.368852
    2       female    0.921053
            male      0.157407
    3       female    0.500000
            male      0.135447
    Name: Survived, dtype: float64


Turns out that the predications stays the same, with or without the Pclass prediactor. Logistic Regression computes the probabilities for each response and predicts the response with largest likelihood. In this case, as we can see from the above pivot table, the most probable survival response for female is 1 and 0 for male, despite the change in survival rate due to Pclass, which is why the predictions stay the same.  

Even though we didn't improve the model with the addition of the Pclass predicators, our effort was not totally worthless. Now we can add more variables to the Logistic Regression, hoping it would produce a more accurate prediction with the extra data. In the next post, we will explore some more complicated models. 