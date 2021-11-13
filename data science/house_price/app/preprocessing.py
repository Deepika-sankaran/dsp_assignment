from sklearn import linear_model
from sklearn.linear_model import LinearRegression

from app.train import X_train, y_train, X, y

regr = LinearRegression()
regr.fit(X_train, y_train)

regr = linear_model.LinearRegression()
regr.fit(X, y)
predictedSalePrice = regr.predict([[5, 20]])
print(predictedSalePrice)
