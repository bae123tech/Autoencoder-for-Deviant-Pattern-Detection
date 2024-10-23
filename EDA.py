# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure plot styles and sizes
plt.style.use('ggplot')
sns.set_palette('pastel')
plt.rcParams['figure.figsize'] = (10, 6)

# Load and preview dataset (adjust path as needed)
dataset_path = '/Users/triveshsharma/Desktop.csv'
df = pd.read_csv(dataset_path)

# Display first 5 rows and data summary
print("First few records of the dataset:")
print(df.head())
print("\nDataset structure and types:")
print(df.info())

# Check for missing values and basic statistics
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nSummary statistics of numerical features:")
print(df.describe())

# Identify numerical and categorical features
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
print("\nNumerical Columns:", numerical_columns)
print("Categorical Columns:", categorical_columns)

# Perform Exploratory Data Analysis (EDA)
# Pie chart for class distribution (Fraud vs Non-fraud)
class_distribution = df['Class'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(class_distribution, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'red'])
plt.title('Fraud vs Non-Fraud Transactions')
plt.show()

# Correlation heatmap for all numerical features
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Distribution of specific numerical features
features_to_plot = ['V1', 'V2', 'V3', 'Amount']
for feature in features_to_plot:
    plt.figure()
    sns.histplot(df[feature], bins=50, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()

# Scaling numerical features (no categorical variables in this dataset)
scaler = StandardScaler()
scaled_features = pd.DataFrame(scaler.fit_transform(df[numerical_columns]), columns=numerical_columns)

# Prepare data for anomaly detection (train-test split)
X = scaled_features.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("\nShape of Training Data:", X_train.shape)
print("Shape of Testing Data:", X_test.shape)

# Count genuine (Class 0) and fraudulent (Class 1) samples
genuine_count = df[df['Class'] == 0].shape[0]
fraud_count = df[df['Class'] == 1].shape[0]
print(f"\nGenuine Transactions: {genuine_count}")
print(f"Fraudulent Transactions: {fraud_count}")

# List all columns in the dataset
print("\nDataset Columns:")
print(df.columns)

# Check and plot additional categorical features if they exist
categorical_features_to_plot = ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']

for cat_feature in categorical_features_to_plot:
    if cat_feature in df.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x=cat_feature, hue='Class')
        plt.title(f'{cat_feature} Distribution by Class')
        plt.show()
    else:
        print(f"Feature '{cat_feature}' not found in the dataset.")
