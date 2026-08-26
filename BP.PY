import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    "Age":[20,30,40,50,60],
    "BP":[110,120,130,145,155]
})

model = LinearRegression()

model.fit(df[["Age"]],df["BP"])

print(model.predict([[45]])) 

