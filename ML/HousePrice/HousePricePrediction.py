import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from sklearn import svm

house_data = pd.read_csv("kc_house_data.csv")

labels = house_data['price']
conv_dates = [1 if values == 2014 else 0 for values in house_data.date ]
house_data['date'] = conv_dates
train = house_data.drop(['id', 'price'],axis=1)

x_train , x_test , y_train , y_test = train_test_split(train , labels , test_size = 0.10,random_state =2)

reg = LinearRegression()
reg.fit(x_train,y_train)
print(reg.score(x_test,y_test))

gbr_obj = ensemble.GradientBoostingRegressor(n_estimators = 1000, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')
gbr_obj.fit(x_train, y_train)
gbr_obj.score(x_test,y_test)

model = svm.SVR(kernel='rbf', C=10, gamma=0.0001)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))