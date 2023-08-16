# Import the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 

import warnings
warnings.filterwarnings('ignore')

# Open the file
with open('po1_data.txt', 'r') as file:
    # Read the contents of the file
    data = file.readlines()

# Remove any leading or trailing whitespace from each line
data = [line.strip() for line in data]

# Specify the column names
column_names = ['Subject_no.', 'Jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)', 'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)', 'anhr&hnr', 'nhr', 'hnr', 'meadian_pitch', 'mean_pitch', 'SD_pitch', 'min_pitch', 'max_pitch', 'no.pulse', 'no.period', 'mean_period', 'sd_period', 'frac_UVF', 'NO.VB', 'Deg.VB', 'updrs', 'status']

# Create the DataFrame
df = pd.DataFrame([line.split(',') for line in data], columns=column_names)






df.describe()


df.info()


# Convert columns to float64
float_columns = ['Jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)', 'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)', 'anhr&hnr', 'nhr', 'hnr', 'meadian_pitch', 'mean_pitch', 'SD_pitch', 'min_pitch', 'max_pitch', 'no.pulse', 'no.period', 'mean_period', 'sd_period', 'frac_UVF', 'NO.VB', 'Deg.VB', 'updrs']
df[float_columns] = df[float_columns].astype(float)

# Convert columns to int64
int_columns = ['Subject_no.', 'status']
df[int_columns] = df[int_columns].astype(int)


df.info()


df.describe()


df.shape


df.status.value_counts()


df1 = df.groupby('Subject_no.').mean().reset_index()

#drop the subject no. column from the dataset
df1.drop('Subject_no.', axis=1, inplace=True)

# Compute the correlation matrix
correlation_matrix = df1.corr()

# Get the absolute correlation values with respect to the "status" column
correlation_with_status = correlation_matrix['status'].abs().sort_values(ascending=False)

# Print the salient features
salient_features = correlation_with_status[correlation_with_status > 0.3].index
print(salient_features)


# Plot the correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# Calculate Spearman's rank correlation coefficient
from scipy.stats import spearmanr
correlation_coefficients = {}
for column in df1.columns:
    if column != 'status':
        coefficient, _ = spearmanr(df1[column], df1['status'])
        correlation_coefficients[column] = coefficient

# Sort the correlation coefficients in descending order
sorted_coefficients = sorted(correlation_coefficients.items(), key=lambda x: abs(x[1]), reverse=True)

# Print the correlation coefficients
for column, coefficient in sorted_coefficients:
    print(f"{column}: {coefficient}")


#Create a scatter plot for updrs
plt.scatter(df1['status'], df1['updrs'])
plt.xlabel('status')
plt.ylabel('updrs')
plt.title('Scatter Plot')
plt.show()


#Create a scatter plot Degree of voice breaks
plt.scatter(df1['status'], df1['Deg.VB'])
plt.xlabel('status')
plt.ylabel('Deg.VB')
plt.title('Scatter Plot')
plt.show()


#Create a scatter plot No of voice breaks
plt.scatter(df1['status'], df1['NO.VB'])
plt.xlabel('status')
plt.ylabel('NO.VB')
plt.title('Scatter Plot')
plt.show()


#Attributes in the Group
Atr1='updrs'
Atr2='Deg.VB'
Atr3='NO.VB'


##EDA: Correlation of attributes of group with other attributes.
corr_atr1=df1[df1.columns].corr()[Atr1][:]
corr_atr2=df1[df1.columns].corr()[Atr2][:]
corr_atr3=df1[df1.columns].corr()[Atr3][:]

pd.concat([round(corr_atr1,4),round(corr_atr2,4),round(corr_atr3,4)],axis=1,sort=False).T