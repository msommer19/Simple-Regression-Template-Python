#Data Preprocessing template

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # include all rows (: <- indicates range) and all but the last column (:-1)
y = dataset.iloc[:, -1].values # include only last column (index: -1)
# Machine Learning Programs expect two entities: X and y; not the dataset as a whole!

#Taking care of missing data
# Missing data can lead to errors in ML programs
# e.g. 1% missing data is fine as it does not affect the model learning much
# Here: classic wy to handle missing data: replace missing data with mean value
# Missing values may be replaced with: Median, Max/Min, Mean value of data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # Object of the SimpleImputer class
imputer.fit(X[:, 1:3]) #include only columns with numerical values # Connect X to imputer Object!
# Rule of thumb: Include all numerical columns to impute as in reality you deal with huge matrices where it is not clear if data is complete
# IMPORTANT: upper bound in an array is excluded in Python, hence 1:3 and not 1:2!
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding categorical data
#important: categorical data should not be interpreted as numerical values such that it suggests a numerical order!
#here we generate columns for each categorie and then binary 0/1 for the countries France, Spain, Germany!
# Method called One-Hot-Encoding <- very useful

#Encoding the Independent variable
#How to One-Hot-Encode categorical data:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X  = np.array(ct.fit_transform(X)) # force object to be a numpy array with np.array()

#Encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y  = le.fit_transform(y)

#Splitting datasaet into training and test set
# Test set should not be manipulated! Hence Feature Scaling is always after Splitting the dataset
# The Test set should be "unknown" and by doing feature scaling we grab knowledge from the test set
# To prevent information leakage from the Test set!
# with the test set we evaluate the model on new data!

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =  0.2, random_state = 1) # fixing the seed for education purposes; 0.2 test size is good choice!

#Feature Scaling
# Feature scaling is important to not have some features dominate others
# Goal of feature scaling: Have all features in the same range!
# Hence Dummy or binary variables must not be scaled!!
# Further scaling dummy variables makes them no more interpretable

# So perform Feature Scaling only on numerical values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # performs standardiztation (i.e - mean(x) / sd(x))
# fit method calculats mean and sd
# transform method applies standardisation formula:
X_train[:, 3:]= sc.fit_transform(X_train[:, 3:]) # only numerical values
# for test set only transform to apply the same mean and sd on the new test set:
X_test[:, 3:] = sc.transform(X_test[:, 3:])



