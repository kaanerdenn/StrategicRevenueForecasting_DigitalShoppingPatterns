import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configure display options for Pandas DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.3f}'.format)

# Load and explore the dataset
shop = pd.read_csv("online_shoppers_intention.csv")

def explore_dataframe(dataframe, head=5):
    """Function to provide a general overview of a DataFrame."""
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe())

# Apply the function to explore the dataset
explore_dataframe(shop, head=2)

# Visualize the distribution of the target variable 'Revenue'
plt.figure(figsize=(8, 6))
sns.countplot(data=shop, x='Revenue')
plt.title('Distribution of Revenue')
plt.show()

# Analyzing correlations between features
correlation_matrix = shop.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

# Preparing data for modeling
# Selecting independent variables and the target variable
X = shop.drop(columns=['Revenue'])
y = shop['Revenue']

# Encoding categorical columns
label_encoder = LabelEncoder()
X['Weekend'] = label_encoder.fit_transform(X['Weekend'])

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing and transforming numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
power_transformer = PowerTransformer(method='yeo-johnson')
X_train_transformed = power_transformer.fit_transform(X_train_scaled)
X_test_transformed = power_transformer.transform(X_test_scaled)

# Modeling and Evaluation
# Logistic Regression
lr_model = LogisticRegression(C=0.1)
lr_model.fit(X_train_transformed, y_train)
lr_pred = lr_model.predict(X_test_transformed)
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_conf_matrix = confusion_matrix(y_test, lr_pred)
lr_class_report = classification_report(y_test, lr_pred)
print("Logistic Regression Model:")
print("Accuracy:", lr_accuracy)
print("Confusion Matrix:\n", lr_conf_matrix)
print("Classification Report:\n", lr_class_report)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None)
rf_model.fit(X_train_transformed, y_train)
rf_pred = rf_model.predict(X_test_transformed)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_conf_matrix = confusion_matrix(y_test, rf_pred)
rf_class_report = classification_report(y_test, rf_pred)
print("Random Forest Model:")
print("Accuracy:", rf_accuracy)
print("Confusion Matrix:\n", rf_conf_matrix)
print("Classification Report:\n", rf_class_report)

# CatBoost Classifier
catboost_model = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.1, verbose=0)
catboost_model.fit(X_train_transformed, y_train)

# XGBoost Classifier
xgboost_model = XGBClassifier()
xgboost_model.fit(X_train_transformed, y_train)
xgboost_pred = xgboost_model.predict(X_test_transformed)
xgboost_accuracy = accuracy_score(y_test, xgboost_pred)
xgboost_conf_matrix = confusion_matrix(y_test, xgboost_pred)
xgboost_class_report = classification_report(y_test, xgboost_pred)
print("XGBoost Model:")
print("Accuracy:", xgboost_accuracy)
print("Confusion Matrix:\n", xgboost_conf_matrix)
print("Classification Report:\n", xgboost_class_report)
