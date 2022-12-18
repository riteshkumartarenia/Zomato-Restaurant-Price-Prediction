import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('zomato_df.csv')

df.drop('Unnamed: 0',axis=1,inplace=True)
print(df.head())
x=df.drop('rate',axis=1)
y=df['rate']
print('x:',x.head())
print('Y:',y.head())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=10)


#Preparing Extra Tree Regression
from sklearn.ensemble import  ExtraTreesRegressor, RandomForestRegressor
model_rfr = RandomForestRegressor()
model_rfr.fit(x_train,y_train)


y_predict=model_rfr.predict(x_test)


import pickle
# # Saving model to disk
pickle.dump(model_rfr, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print('Final Output:',y_predict)
