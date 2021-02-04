import analyze
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_squared_error

des, effs = analyze.analyzeOptim("logs/rds.log")

X = np.array(des)
y = np.array(effs)

lr = LinearRegression()
lr.fit(X, y)
ypred = lr.predict(X)

print(lr.coef_)
print(lr.intercept_)
print(r2_score(y, ypred))
print(mean_squared_error(y, ypred))
print(mean_squared_error(y, np.full_like(y, np.mean(y))))

las = Lasso(alpha=0.1)
las.fit(X, y)
ypred = las.predict(X)

print()
print(las.coef_)
print(las.intercept_)
print(r2_score(y, ypred))
print(mean_squared_error(y, ypred))
print(mean_squared_error(y, np.full_like(y, np.mean(y))))
