import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# load dataset
df = pd.read_csv("titanic.csv")

print("First rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# handle missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)

# encode categorical
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# scaling
scaler = StandardScaler()
df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])

# outlier visualization
sns.boxplot(x=df['Fare'])
plt.title("Outliers in Fare")
plt.show()

print("\nCleaned data:")
print(df.head())

# save cleaned data
df.to_csv("cleaned_titanic.csv", index=False)
