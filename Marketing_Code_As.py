#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#import CSV into python environment 
file_path = '/Users/vincent/Desktop/As.2/apple_quality.csv'

df = pd.read_csv(file_path)

df


# In[3]:


df.describe()


# In[4]:


df = df.drop('A_id', axis=1)
df.info()


# In[5]:


# Check for missing values
missing_values_count = df.isnull().sum()

print(missing_values_count)


# In[6]:


# Drop all rows with any missing values
df_cleaned = df.dropna()

df_cleaned.head()


# In[7]:


#check data types of the columns
df.dtypes


# In[8]:


#change the data types of the other columns
df_cleaned['Acidity'] = pd.to_numeric(df_cleaned['Acidity'], errors='coerce')

df_cleaned['Quality_Num'] = df_cleaned['Quality'].map({'good':1,'bad':0})

df_cleaned = df_cleaned.drop(['Quality'], axis=1)

df_cleaned.to_csv('/Users/vincent/Desktop/As.2/apple_quality_clean.csv', index=False)

df_cleaned


# In[9]:


df_cleaned.dtypes


# In[10]:


from scipy.stats import shapiro
import pandas as pd

# Perform Shapiro-Wilk test for each numeric column
for column in df_cleaned.select_dtypes(include=['float64', 'int64']).columns:
    stat, p_value = shapiro(df_cleaned[column].dropna())  # dropna to ignore NaN values for the test
    print(f'Column: {column}, Statistics={stat:.4f}, p-value={p_value:.4g}')
    # Interpretation
    alpha = 0.05
    if p_value > alpha:
        print(f'  {column} looks like it follows a normal distribution (fail to reject H0)\n')
    else:
        print(f'  {column} does not look like it follows a normal distribution (reject H0)\n')


# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 12))

plt.subplot(4, 2, 1)
sns.histplot(x=df['Size'])
plt.title('Histogram of Size')

plt.subplot(4, 2, 2)
sns.histplot(x=df['Sweetness'])
plt.title('Histogram of Sweetness')

plt.subplot(4, 2, 3)
sns.histplot(x=df['Crunchiness'])
plt.title('Histogram of Crunchiness')

plt.subplot(4, 2, 4)
sns.histplot(x=df['Ripeness'])
plt.title('Histogram of Ripeness')

plt.subplot(4, 2, 5)
sns.boxplot(x=df['Size'])
plt.title('Box-plot of Size')

plt.subplot(4, 2, 6)
sns.boxplot(x=df['Sweetness'])
plt.title('Box-plot of Sweetness')

plt.subplot(4, 2, 7)
sns.boxplot(x=df['Crunchiness'])
plt.title('Box-plot of Crunchiness')

plt.subplot(4, 2, 8)
sns.boxplot(x=df['Ripeness'])
plt.title('Box-plot of Ripeness')

plt.tight_layout()
plt.show()


# In[12]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# Compute the correlation matrix
corr = df_cleaned.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(8, 6))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.show()


# In[13]:


sns.pairplot(df_cleaned, hue='Quality_Num')
plt.figure(figsize= (10, 6))
plt.show()


# In[14]:


#Quality predeiction with all measuers 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Assuming 'Quality_Num' is the target and is a separate column in your DataFrame
# and all other columns are features
X = df_cleaned.drop('Quality_Num', axis=1)
y = df_cleaned['Quality_Num']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the Random Forest Classifier: {accuracy:.2f}')


# In[15]:


# Extract feature importances
feature_importances = rf_classifier.feature_importances_

# Create a pandas series to make it easier to visualize the importances
features_series = pd.Series(feature_importances, index=X.columns).sort_values(ascending=False)

# Display the feature importances
print("Feature Importances:")
print(features_series)


# In[16]:


#only using measures which you can determin without cutting into the apple
# Selecting predictors
X = df_cleaned[['Size', 'Weight', 'Ripeness']]
y = df_cleaned['Quality_Num']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model with your selected features
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the Random Forest Classifier: {accuracy:.2f}')


# In[17]:


# Get feature importances
importances = rf_classifier.feature_importances_

# Summarize feature importances
for feature, importance in zip(['Size', 'Weight', 'Ripeness'], importances):
    print(f'Feature: {feature}, Importance: {importance:.4f}')


# In[18]:


quality_means = df_cleaned.groupby('Quality_Num').mean()
print(quality_means)


# In[19]:


df_filtered = df_cleaned[df_cleaned['Quality_Num'] == 0]

df_filtered.describe()


# In[20]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

features = df_filtered.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_filtered[features])

# Perform K-Means clustering with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
df_filtered['Cluster'] = kmeans.fit_predict(X_scaled)

# Display the first few rows to confirm clustering
df_filtered.head()


# In[21]:


quality_means = df_filtered.groupby('Cluster').mean()
print(quality_means)


# In[22]:


df_cleaned_merged = df_cleaned.merge(df_filtered[['Cluster']], left_index=True, right_index=True, how='left')

df_cleaned_merged['Cluster'].fillna(2, inplace=True)

df_cleaned_merged


# In[23]:


quality_means = df_cleaned_merged.groupby('Cluster').mean()
print(quality_means)


# In[39]:


import matplotlib.pyplot as plt

# Calculate the percentage of the clusters
cluster_counts = df_cleaned_merged['Cluster'].value_counts(normalize=True) * 100
custom_labels = ['Balanced and Versatile', 'Sweet and Soft', 'Tangy and Crunchy']
colors = ['Crimson', 'LightSalmon', 'LightCoral']

plt.figure(figsize=(8, 6))
plt.pie(cluster_counts, labels = custom_labels, autopct='%1.1f%%', startangle=140, colors=colors)
plt.axis('equal')  

plt.savefig('/Users/vincent/Desktop/3-cluster_percentage_pie_chart.jpeg', format='jpeg', dpi=300)

plt.show()


# In[40]:


cluster_counts = df_cleaned['Quality_Num'].value_counts(normalize=True) * 100
own_labels = ['Good Quality', 'Bad Quality']
colors = ['Crimson', 'Coral']

plt.figure(figsize=(8, 6))
plt.pie(cluster_counts, labels = own_labels, autopct='%1.1f%%', startangle=140, colors=colors)
plt.axis('equal')  

plt.savefig('/Users/vincent/Desktop/2-cluster_percentage_pie_chart.jpeg', format='jpeg', dpi=300)

plt.show()


# In[ ]:




