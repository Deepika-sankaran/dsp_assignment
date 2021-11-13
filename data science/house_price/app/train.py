import numpy as np
import pandas as pd

from app.preprocessing import regr

dataset = pd.read_csv(train_filepath)

def train(train_filepath):
    dataset = pd.read_csv(train_filepath)
    return dataset

def check_null():
    asnull = dataset.isnull().sum()
    return asnull

def duplicates():
    dupl = dataset.duplicated().sum()
    return dupl

#removing blank rows
def rem_unwanted():
    remv = dataset.drop("Alley", axis=1)
    return remv

def num_data():
    numeric_data = dataset.select_dtypes(include=[np.number])
    var = numeric_data.shape[1]
    return numeric_data

def cat_data():
    categorical_data = dataset.select_dtypes(exclude=[np.number])
    var = categorical_data.shape[1]
    return categorical_data

X = dataset[['Id', 'MSSubClass']]
y = dataset['SalePrice'].values.reshape(-1, 1)

# Train-test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# model accuracy
y_pred = regr.predict(X_test)
print("Residual sum of squares (RSS): %.2f" % sum((y_pred - y_test) ** 2))
print("Mean absolute error (MAE): %.2f" % pd.np.mean(abs(y_pred - y_test)))
print("Mean square error (MSE): %.2f" % pd.np.mean((y_pred - y_test) ** 2))
print("Root mean square error (RMSE): %.2f" % pd.np.sqrt(pd.np.mean((y_pred - y_test) ** 2)))





