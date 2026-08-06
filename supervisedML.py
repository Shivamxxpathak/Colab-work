import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

#Dataset
df = pd.DataFrame({
    "Hours":[1,2,3,4,5,6,7,8,9,10],
    "Marks":[10,20,30,40,50,60,70,80,90,100]
})

X  = df[['Hours']]
y  = df['Marks']

#Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Train
model = LinearRegression()
model.fit(X_train,y_train)

#Predict 
y_pred = model.predict(X_test)
print("Predicted:",y_pred)

#Plot
plt.scatter(X,y,color = 'blue')
plt.plot(X,model.predict(X),color = 'red')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.title('Linear Regression Example')
plt.show()

#evaluation
print('MSE:',mean_squared_error(y_test,y_pred))
print('RMSE:',np.sqrt(mean_squared_error(y_test,y_pred)))
print('R2Score:',r2_score(y_test,y_pred))