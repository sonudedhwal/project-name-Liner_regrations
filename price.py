import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "Size":[1000,1200,1500,1800,2000],
    "Price":[150000,180000,220000,260000,300000]
}

df = pd.DataFrame(data)

X = df[["Size"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

predicted_price = model.predict([[1700]])

print("Predicted Price:", predicted_price[0]) 


