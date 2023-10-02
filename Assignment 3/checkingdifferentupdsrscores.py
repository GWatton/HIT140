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
timeadjust = highr2group[highr2group['test_time'] > 0]

lowerupdrs = timeadjust[timeadjust['motor_updrs'] >0]
upperupdrs = lowerupdrs[lowerupdrs['motor_updrs'] <60]

# Plot correlation matrix
corr = upperupdrs.corr()

# Plot the pairwise correlation as heatmap
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=False,
    annot=True
)

# customise the labels
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

plt.show()

xinput = upperupdrs[['jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']]

x = upperupdrs[[ 'jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']]
y = upperupdrs[['motor_updrs']]

# Build and evaluate the linear regression model
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)

# build and evaluate the linear regression model
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)


"""
APPLY Z-SCORE STANDARDISATION
"""
scaler = StandardScaler()

# Drop the previously added constant
x = x.drop(["const"], axis=1)

# Apply z-score standardisation to all explanatory variables
std_x = scaler.fit_transform(x.values)
print('bananas', std_x)


# Restore the column names of each explanatory variable
std_x_df = pd.DataFrame(std_x, index=x.index, columns=x.columns)



"""
REBUILD AND REEVALUATE LINEAR REGRESSION USING STATSMODELS
USING STANDARDISED EXPLANATORY VARIABLES
"""

# Build and evaluate the linear regression model
std_x_df = sm.add_constant(std_x_df)

print(std_x_df)
model = sm.OLS(y,std_x_df).fit()
pred = model.predict(std_x_df)
model_details = model.summary()
print(model_details)

x = upperupdrs[['jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']]
y = upperupdrs.iloc[:,4].values

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


"""
REBUILD AND REEVALUATE LINEAR REGRESSION USING STATSMODELS
WITH COLLINEARITY BEING FIXED
"""

# Drop one or more of the correlated variables. Keep only one.
# df = df.drop(["RAD"], axis=1)
# print(df.info())

# Separate explanatory variables (x) from the response variable (y)
# x = df.iloc[:,:-1]
# y = df.iloc[:,-1]

# # Build and evaluate the linear regression model
# x = sm.add_constant(x)
# model = sm.OLS(y,x).fit()
# pred = model.predict(x)
# model_details = model.summary()
# print(model_details)