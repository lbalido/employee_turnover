# Employee Turnover

## Overview
Caregiver turnover rate was 39.4% in 2009, increased to 65.7% in 2017, and is at an all time high of 82% in 2018. Some of the reasons for the high turnover rate is low pay (median is $11.00/hr across the US), no benefits such as paid time off, and limited job security.

Not only is it getting harder to keep employees, but the demand for caregivers is increasing due to the baby boomer generation.

The Center for American Progress did a study that said the average cost of replace any employee that earns less than $30,000 per year is 16% of their annual salary. The median pay rate for my specific dataset is $8.00/hr.  This means that it costs the company $2,000 to replace any employee. These hidden costs are due to recruiting and training, loss of work quality, and fewer employee referrals.

## Gathering the Data

This information is taken from a real home care copmany for the dates of January 2013 through the end of July 2019. The data was taken from two data sets.  One with payroll information that included shifts worked, pay rates, hours, etc. This had 482,580 rows with 13 features. The other with employee basic information such as address, age, hire date, etc. This had 7,738 employees with 11 features.



## Features

After the first round of EDA, many of the features were not important. I need more features in order to create a better model. I took out features such as gender and state because they had little importance.

![not_important](images/not_important.png?raw=true "not_important")


This is when I collected pay data from the payroll database. I grouped them by employee and merged it with my previous dataset.


## EDA 

I was originally going to do active vs inactive employees, then realized after I added all the data together that almost all of the employees were inactive because it's such a high turnover industry and the data is over a few years. I decided to change my project to classify based on length of employment.

In order to choose the amount of time that would be the cut off point for classification, I looked at the distribution of the amount of days the employees were with the company. It was very skewed, so instead of using the mean value, I chose to use the median.  The median ended up being 72 days. 

Going forward I will refer to "active" as employees who stay with the company for more than 72 days and "inactive" as employess who stay with the company for less than 72 days.

![median](images/median.png?raw=true "median")



## Models 

Since I've split the data from active vs inactive at the median amount of days of employment length, without a model there is a 50% chance that the they will stay for longer than 72 days. 

I used a grid search for each model in order to find the best parameters.

Model: LogisticRegression
precision = 67.04%
recall = 59.93%
accuracy = 65.62%, score = 65.62%

Model: GradientBoostingClassifier
precision = 70.83%
recall = 74.28%
accuracy = 72.16%, score = 72.16%

Model: RandomForestClassifier
precision = 72.07%
recall = 70.54%
accuracy = 71.91%, score = 71.91%

Model: KNeighborsClassifier
precision = 64.72%
recall = 59.55%
accuracy = 63.95%, score = 63.95%

Model: DecisionTreeClassifier
precision = 67.88%
recall = 72.03%
accuracy = 69.32%, score = 69.32%

## Feature Importance

These two graphs are based on the feature importance for Gradient Boosting. 
![Feature Importance](images/feature_importance.png?raw=true "feature_importance")
![Feature Importance2](images/feature_importance2.png?raw=true "feature_importance2")


You can see here that the most important features are the if they work weekdays, the type of client, and the age.  
![features_vs_eda](images/features_vs_eda.png?raw=true "features_vs_eda")

I wanted to look into the pay rate feature a little more closely to see what the ideal pay rate would be.
![pay_rate](images/pay_rate.png?raw=true "pay_rate")
![pay_rate_percent](images/pay_rate_percent.png?raw=true "pay_rate_percent")


## Recommendations:

I would recommend to raise the pay rate based on the active percentage of caregivers. I think a good rate would be $8.50-$9.00/hr.

I would also recommend to look at the different types of clients and see if they have  
