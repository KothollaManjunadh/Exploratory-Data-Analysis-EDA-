import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = sns.load_dataset('iris')

# Basic info
print(df.head())
print(df.info())
print(df.describe())

# Missing values
print(df.isnull().sum())

# Histogram
plt.figure(figsize=(8,5))
plt.hist(df['sepal_length'], bins=10)
plt.title("Sepal Length Distribution")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()

# Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(y=df['petal_length'])
plt.title("Petal Length Boxplot")
plt.show()

# Scatter plot
plt.figure(figsize=(7,5))
sns.scatterplot(
    x='sepal_length',
    y='petal_length',
    hue='species',
    data=df
)

plt.title("Sepal Length vs Petal Length")
plt.show()

# Bar plot
plt.figure(figsize=(7,5))
sns.barplot(
    x='species',
    y='sepal_length',
    data=df
)

plt.title("Average Sepal Length by Species")
plt.show()

# Correlation matrix
corr = df.corr(numeric_only=True)

print(corr)

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df, hue='species')

plt.show()

print("EDA Completed Successfully")
