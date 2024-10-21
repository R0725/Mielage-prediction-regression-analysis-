import pandas as pd

# Load the Auto MPG dataset from an alternative URL
url = "https://raw.githubusercontent.com/plotly/datasets/master/mpg.csv"

# Read the dataset
data = pd.read_csv(url)

# Display the first few rows and check for missing values
print(data.head())
print("\nMissing values in each column:")
print(data.isnull().sum())

# Drop rows with missing values or you can fill them
data.dropna(inplace=True)  # Drop rows with missing values

# Alternatively, you can fill missing values (comment out the above line if using this):
# data['horsepower'].fillna(data['horsepower'].mean(), inplace=True)

# Now define features and target variable
X = data[['horsepower', 'weight']]
y = data['mpg']

# Import necessary libraries for regression analysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')

# Visualize the results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Actual vs. Predicted MPG')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.show()
