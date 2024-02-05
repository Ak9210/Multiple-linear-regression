import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


data = pd.read_csv('data.csv')


X = data[['Volume', 'Weight']]
y = data['CO2']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


sorted_indices = np.argsort(y_test)
y_test_sorted = np.array(y_test)[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

plt.plot(y_test_sorted, label='Actual CO2 Values')
plt.plot(y_pred_sorted, label='Predicted CO2 Values')
plt.xlabel("Data Points")
plt.ylabel("CO2 Values")
plt.title("Actual vs. Predicted CO2 Values in Multiple Linear Regression with Connecting Lines")
plt.legend()
plt.show()


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
