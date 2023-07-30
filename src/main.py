# Predict text messages as “spam” or “not spam” using 3 ML models
# Dataset used: https://github.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/blob/master/spam.csv

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Cleaning the raw data
data = pd.read_csv('spam.csv')
data.rename(columns={"v1": "label", "v2": "text"}, inplace=True)
data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1, inplace=True)
data.replace({'ham': 0, 'spam': 1}, inplace=True)

# Split data into training and test data
X = TfidfVectorizer().fit_transform(data['text']).toarray()
y = data['label']
# 4:1 for training and test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Trivial model
print("Accuracy for trivial model: ", accuracy_score(y, np.zeros_like(y)))

# Linear Regression
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.score(x_test, y_test)
print("Training accuracy for linear regression: ", predictions)
kfold = model_selection.KFold(n_splits=10)
results_kfold = model_selection.cross_val_score(model, X, y, cv=kfold)
print("Validation accuracy for linear regression: ", results_kfold.mean()) 

# Logistic Regression
model = LogisticRegression()
model.fit(x_train, y_train)
predictions = model.score(x_test, y_test)
print("Training accuracy for logistic regression: ", predictions)
kfold = model_selection.KFold(n_splits=10)
results_kfold = model_selection.cross_val_score(model, X, y, cv=kfold)
print("Validation accuracy for logistic regression: ", results_kfold.mean()) 

# Naive Bayes
model = GaussianNB()
model.fit(x_train, y_train)
predictions = model.score(x_test, y_test)
print("Training accuracy for naive bayes: ", predictions)
kfold = model_selection.KFold(n_splits=10)
results_kfold = model_selection.cross_val_score(model, X, y, cv=kfold)
print("Validation accuracy for naive bayes: ", results_kfold.mean()) 