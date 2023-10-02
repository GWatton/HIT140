import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PowerTransformer

"""
BUILD AND EVALUATE LINEAR REGRESSION USING STATSMODELS
"""

# Read dataset into a DataFrame
df = pd.read_csv('po2_data.csv')
column_names = ['subject#','age','sex','test_time','motor_updrs','total_updrs','jitter_percent', 'jitter_abs', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer_percent', 'shimmer_abs', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']
df.columns= column_names

groupin30s = df[df["subject#"].isin([32])]
groupin40s = df[df["subject#"].isin([26,30])]
groupin50s = df[df["subject#"].isin([2,3,10,11,14,19,22,23, 27,34,37])]
groupin60s = df[df["subject#"].isin([6,9,12,15,16,17,18,20,24,33,36,38,39,41,42])]
groupin70s = df[df["subject#"].isin([1,4,5,7,8,13,21,25,28,29,31,35])]
groupin80s = df[df["subject#"].isin([40])]



# group1= df[df["subject#"].isin([ 2, 4,7,8,16,20,23,24,27,29,36,37,38,42])]
group1= df[df["subject#"].isin([2,4,20,29,37])]


timeadjust = group1[group1['test_time'] > 90]


wtf = group1[['jitter_percent', 'jitter_abs', 'jitter_rap',  'jitter_ddp',  'shimmer_apq5', 'shimmer_apq11', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']]
wtf2 = group1[['total_updrs']]
# Separate explanatory variables (x) from the response variable (y)
x = wtf
y = wtf2

# Reparate explanatory variables (x) from the response variable (y)

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
