import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
    "Hours":[1,2,3,4,5,6,7,8],
    "Score":[2,4,5,6,7,8,8.6,9]
}
df = pd.DataFrame(data)

# df.plot(x="Hours", y="Score", style='o')


X = df[["Hours"]]
Y = df["Score"]

model = LinearRegression()

model.fit(X,Y)

new_hours = [[6]]

predicted_score = model.predict(new_hours)

print("Predicted score: ", predicted_score[0])

new_data = [[4],[6],[9]]

predictions = model.predict(new_data)

print(predictions)


plt.scatter(X,Y)
plt.plot(X, model.predict(X))
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.title('Hours vs Score')
plt.show()


from sklearn.metrics import r2_score
y_pred  = model.predict(X)
score = r2_score(Y, y_pred)
print("R2 score:", score)
