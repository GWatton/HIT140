import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


df = pd.read_csv('po2_data.csv')
column_names = ['subject#','age','sex','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']
df.columns= column_names

#Grouped by age
groupin30s = df[df["subject#"].isin([32])]
groupin40s = df[df["subject#"].isin([26,30])]
groupin50s = df[df["subject#"].isin([2,3,10,11,14,19,22,23, 27,34,37])]
groupin60s = df[df["subject#"].isin([6,9,12,15,16,17,18,20,24,33,36,38,39,41,42])]
groupin70s = df[df["subject#"].isin([1,4,5,7,8,13,21,25,28,29,31,35])]
groupin80s = df[df["subject#"].isin([40])]

##groupedbygender
male = df[df["sex"]==0]
maleredefined = df[df["subject#"].isin([1,2,3,5,6,8,10,11,15,16,19,20,21,24,25,26,29,38,39,42])] 


female = df[df["sex"]==1]
femalerefined= df[df["subject#"].isin([8,13,14,22,23,27,28,33,36,40,41])]


#Best Fit groups
group1= df[df["subject#"].isin([2,4,7,8,16,20,23,24,27,29,36,37,38,42])]


#Highest R2 Group
highr2group= df[df["subject#"].isin([2,4,20,29,37])]

#Use this to adjust for time or remove negative time values
timeadjust = group1[group1['test_time'] > 0]

overallvalues = female[['jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']]

#jitter_percent, jitter_rap, shimmer_abs, shimmer_apq3, and rpde
bestvaluesforgroup1 = group1[['jitter_percent', 'jitter_abs', 'jitter_ppq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']]

bestvaluesforfemales = femalerefined[[ 'jitter_rap',  'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq5', 'shimmer_apq11', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']]

bestvaluesformales = maleredefined[[ 'jitter_abs', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr',  'dfa', 'ppe']]

bestfitforhighr2group = highr2group[['jitter_percent', 'jitter_abs', 'jitter_rap',  'jitter_ddp',  'shimmer_apq5', 'shimmer_apq11', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']]


#Used to input Group1 detailes
# wtf = group1[['jitter_percent', 'jitter_abs', 'jitter_rap',  'jitter_ddp',  'shimmer_apq5', 'shimmer_apq11', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']]
# wtf2 = group1[['total_updrs']]
onlycolinear = group1[['dfa']]

wtf2=femalerefined[['total_updrs']]


# Reparate explanatory variables (x) from the response variable (y)
x = bestvaluesforfemales
y = wtf2

# Build and evaluate the linear regression model
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)


"""
APPLY POWER TRANSFORMER
"""

# Drop the previously added constant
x = x.drop(["const"], axis=1)

# Create a Yeo-Johnson transformer
scaler = PowerTransformer()

# Apply the transformer to make all explanatory variables more Gaussian-looking
std_x = scaler.fit_transform(x.values)

# Restore column names of explanatory variables
std_x_df = pd.DataFrame(std_x, index=x.index, columns=x.columns)


"""
REBUILD AND REEVALUATE LINEAR REGRESSION USING STATSMODELS
USING MORE GAUSSIAN-LIKE EXPLANATORY VARIABLES
"""

# Build and evaluate the linear regression model
std_x_df = sm.add_constant(std_x_df)
model = sm.OLS(y,std_x_df).fit()
pred = model.predict(std_x_df)
model_details = model.summary()
print(model_details)

x = std_x
y = femalerefined.iloc[:,4].values

print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()

# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred))
# Normalised Root Mean Square Error
y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test, y_pred)

print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)



"""
COMPARE THE PERFORMANCE OF THE LINEAR REGRESSION MODEL
VS.
A DUMMY MODEL (BASELINE) THAT USES MEAN AS THE BASIS OF ITS PREDICTION
"""

# Compute mean of values in (y) training set
y_base = np.mean(y_train)

# Replicate the mean values as many times as there are values in the test set
y_pred_base = [y_base] * len(y_test)


# Optional: Show the predicted values of (y) next to the actual values of (y)
df_base_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_base})
print(df_base_pred)

# Compute standard performance metrics of the baseline model:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test, y_pred_base)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred_base)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred_base))

# Normalised Root Mean Square Error
y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test, y_pred_base)

print("Baseline performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)