import pandas as pd
from sklearn import linear_model

from app.preprocessing import regr
from app.train import dataset


def inference(test_filepath, MODELS_DIR):
    dataset = pd.read_csv(test_filepath)
    return dataset

X_test = pd.read_csv('test.csv')
X_test = X_test.loc[:, ['MSSubClass', 'LotArea']]
regr.predict(X_test)




