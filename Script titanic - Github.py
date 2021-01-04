import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Loading the passenger data
passengers = pd.read_csv('passengers.csv')

# Updating sex column to numerical (1 for female and 0 for men)
passengers['Sex'] = passengers.Sex.apply(lambda x:1 if x=='female' else 0)

# Filling the nan values in the age column by the mean value
passengers['Age'] = passengers.Age.fillna(value = 29.6)

# Creating a first class column (1 if first class, 0 otherwise)
passengers['FirstClass'] = passengers.Pclass.apply(lambda x:1 if x==1 else 0)

# Creating a second class column (1 if first class, 0 otherwise)
passengers['SecondClass'] = passengers.Pclass.apply(lambda x:1 if x==2 else 0)

# Selecting the features - Sex, Age, FirstClass, SecondClass
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]

survival = passengers['Survived']

# Performing train, test, split
features_train, features_test, survival_train, survival_test = train_test_split(features, survival, train_size = 0.8, test_size = 0.2)

# Scaling the feature data 
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Creating and training the model. 
model = LogisticRegression()
model.fit(features_train, survival_train)

# Scoring the model on the train data. 
training_score = model.score(features_train, survival_train)
print(training_score)

# Scoring the model on the test data. 
test_score = model.score(features_test, survival_test)
print(test_score)

# Analyzing the coefficients determined by the model. 
coefficients = model.coef_
print(coefficients)

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Me = np.array([1.0,36,0.0,1.0])
Husband = np.array([0.0, 37,0.0,1.0])
Kid1 = np.array([0.0, 5, 0.0,1.0])
Kid2 = np.array([0.0, 2.5, 0.0, 1.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, Me, Husband, Kid1, Kid2])

# Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)
print(sample_passengers)

# Make survival predictions!
prediction = model.predict(sample_passengers)
print(prediction)

prediction_proba = model.predict_proba(sample_passengers)
print(prediction_proba)