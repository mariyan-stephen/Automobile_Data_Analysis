import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
column_names = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
df = pd.read_csv(url, delimiter='\s*,\s*', names=column_names, engine='python')

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Handle missing values
df['normalized-losses'] = pd.to_numeric(df['normalized-losses'], errors='coerce')
df['normalized-losses'].fillna(df['normalized-losses'].mean(), inplace=True)

df['num-of-doors'].fillna(df['num-of-doors'].mode()[0], inplace=True)

df['bore'] = pd.to_numeric(df['bore'], errors='coerce')
df['bore'].fillna(df['bore'].mean(), inplace=True)

df['stroke'] = pd.to_numeric(df['stroke'], errors='coerce')
df['stroke'].fillna(df['stroke'].mean(), inplace=True)

df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)

df['peak-rpm'] = pd.to_numeric(df['peak-rpm'], errors='coerce')
df['peak-rpm'].fillna(df['peak-rpm'].mean(), inplace=True)

df.dropna(subset=['price'], inplace=True)

# Perform data manipulation and feature engineering
cylinder_mapping = {
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'eight': 8,
    'twelve': 12
}
df['num-of-cylinders'] = df['num-of-cylinders'].map(cylinder_mapping)

print(df)

# Data analysis
print(df.describe())

# Data visualization
numeric_columns = df.select_dtypes(include=[np.number])
corr_matrix = numeric_columns.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

# Data modeling
X = df.drop(['price', 'make'], axis=1)
y = df['price'].astype(float)
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Set plot size and font size
plt.figure(figsize=(16, 12))  # You can adjust these values (16, 12) to change the plot size
sns.set(font_scale=0.8)  # You can adjust this value to change the font size

# Generate a heatmap for the correlation matrix
heatmap_plot = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f', linewidths=.5)

# Save the heatmap as an image
plt.savefig('heatmap_plot.png', dpi=300, bbox_inches='tight')

# Print Mean Squared Error and R-squared values
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Display the plot in the console
plt.show()
