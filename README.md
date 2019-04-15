# LRegression

#Imports
Linear Regression Python code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Get the Data
from sklearn.datasets import load_boston
boston_dataset = load_boston()
print(boston_dataset.DESCR)
boston_df = boston_dataset.data

type(boston_df) 
print(boston.keys())

dict_keys=(['data', 'target', 'feature_names', 'DESCR', 'filename'])
boston.feature_names

boston=pd.DataFrame(boston_df,columns=boston.feature_names)
boston.head()

boston['MEDV']=boston_dataset.target

#missing_values
boston.isnull().sum()

#Exploratory Data Analysis (EDA)
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()
sns.pairplot(boston)
sns.heatmap(boston.corr(),annot=True)
#The correlation coefficient ranges from -1 to 1. If the value is close to 1, 
#it means that there is a strong positive correlation between the two variables. 
#When it is close to -1, the variables have a strong negative correlation.

plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')

#Training a linear Regression Model

boston.columns
X=boston[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT', 'MEDV']]
y=boston['MEDV']

#train test split 
from sklearn.model_selection import train_test_split
train_test_split
#prese sfift +Tab and copy paste 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
#test suze : percentage of your data set that you want to be allocated to the test size (40% or 30%)
#random_state: ensures a specific set of random splits

#Creating and Training the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

#Model Evaluation
print(lm.intercept_)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
coeff_boston=pd.DataFrame(lm.coef_,X.columns,columns=['Coefficients'])
coeff_boston

#Prediction from our Model
predictions=lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot(y_test-predictions,bins=50)

#Regression Evaluation Metrics
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
r2 = r2_score(y_test, predictions)

