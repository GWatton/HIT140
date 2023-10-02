import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
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

bestvaluesforfemales = femalerefined[['jitter_abs',  'shimmer_percent', 'shimmer_abs', 'shimmer_apq5', 'shimmer_apq11', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']]

bestvaluesformales = maleredefined[['jitter_abs',  'jitter_ddp', 'shimmer_apq11',  'nhr', 'hnr',  'dfa', 'ppe']]

bestfitforhighr2group = highr2group[['jitter_percent', 'jitter_abs', 'jitter_rap',  'jitter_ddp',  'shimmer_apq5', 'shimmer_apq11', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']]


#Used to input Group1 detailes
# wtf = group1[['jitter_percent', 'jitter_abs', 'jitter_rap',  'jitter_ddp',  'shimmer_apq5', 'shimmer_apq11', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe']]
# wtf2 = group1[['total_updrs']]
onlycolinear = group1[['shimmer_abs']]

wtf2=femalerefined[['total_updrs']]


# Apply non-linear transformation
bestvaluesforfemales["LOGshimmer_percent"] = bestvaluesforfemales["shimmer_abs"].apply(np.log)


# Visualise the effect of the transformation
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.scatter(bestvaluesforfemales["shimmer_abs"], femalerefined["total_updrs"], color="green")
plt.title("Original shimmer_abs")
plt.xlabel("shimmer_abs")
plt.ylabel("total_updrs")
plt.plot([0,40],[30,0])

plt.subplot(1,2,2)
plt.scatter(bestvaluesforfemales["LOGshimmer_percent"], femalerefined["total_updrs"], color="red")
plt.title("Log Transformed shimmer_abs")
plt.xlabel("LOGshimmer_percent")
plt.ylabel("total_updrs")
plt.plot([0,4],[50,0])

plt.show()


"""
RE-RUN THE LINEAR REGRESSION MODEL WITH A TRANSFORMED VARIABLE
"""
# Drop the original LSTAT variable
df4 = bestvaluesforfemales.drop("shimmer_abs", axis=1)

# Separate explanatory variables (x) from the response variable (y)
x = df4
y = wtf2

# Build and evaluate the linear regression model
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)


"""
SINCE LOGSTAT WORKS BETTER THAN LSTAT FOLLOWING THE LINEAR TRANSFORMATION:
1. WRITE THE TRANSFORMED DATAFRAME INTO A .CSV FILE
2. RE-RUN THE LINEAR REGRESSION MODEL USING SKLEARN
"""

# # write dataframe to .csv
# df.to_csv("boston_logstat.csv", index=False)