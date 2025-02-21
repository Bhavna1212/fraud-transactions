import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of transactions
num_transactions = 10000

# Generate transaction IDs
transaction_id = np.arange(1, num_transactions + 1)

# Generate user IDs (1000 unique users)
user_id = np.random.randint(1, 1001, num_transactions)

# Generate transaction amounts (fraudulent transactions will have higher amounts)
transaction_amount = np.random.exponential(scale=100, size=num_transactions)
transaction_amount = np.round(transaction_amount, 2)  # Round to 2 decimal places

# Generate transaction types
transaction_types = ["Cash Withdrawal", "Online Payment", "Direct Deposit", "Wire Transfer", "Check Deposit"]
transaction_type = np.random.choice(transaction_types, num_transactions)

# Generate locations (10 different cities)
locations = ["New York", "Los Angeles", "Chicago", "Houston", "Miami", "San Francisco", "Seattle", "Denver", "Boston", "Dallas"]
location = np.random.choice(locations, num_transactions)

# Generate timestamps (random dates in the last 2 years)
timestamps = pd.date_range(start="2023-01-01", periods=num_transactions, freq="H")
timestamps = np.random.choice(timestamps, num_transactions)

# Create fraud labels (5% fraud cases)
fraud_flag = np.zeros(num_transactions)
fraud_indices = np.random.choice(num_transactions, int(0.05 * num_transactions), replace=False)
fraud_flag[fraud_indices] = 1

# Create DataFrame
df = pd.DataFrame({
    "transaction_id": transaction_id,
    "user_id": user_id,
    "transaction_amount": transaction_amount,
    "transaction_type": transaction_type,
    "location": location,
    "timestamp": timestamps,
    "fraud_flag": fraud_flag
})

# Save dataset as CSV
df.to_csv("social_security_fraud_data.csv", index=False)

# Display first 5 rows
df.head()
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("social_security_fraud_data.csv")

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Summary statistics
print("\nSummary Statistics:\n", df.describe())

# Count of fraud vs. non-fraud transactions
plt.figure(figsize=(6, 4))
sns.countplot(x="fraud_flag", data=df, palette="coolwarm")
plt.title("Fraud vs. Non-Fraud Transactions")
plt.xticks([0, 1], ["Non-Fraud", "Fraud"])
plt.show()

# Transaction amount distribution by fraud status
plt.figure(figsize=(8, 5))
sns.boxplot(x="fraud_flag", y="transaction_amount", data=df, palette="coolwarm")
plt.title("Transaction Amount Distribution by Fraud Status")
plt.xticks([0, 1], ["Non-Fraud", "Fraud"])
plt.show()

# Transactions by type and fraud status
plt.figure(figsize=(10, 5))
sns.countplot(y="transaction_type", hue="fraud_flag", data=df, palette="coolwarm")
plt.title("Transaction Type Distribution by Fraud Status")
plt.legend(["Non-Fraud", "Fraud"])
plt.show()

# One-hot encoding categorical variables
df = pd.get_dummies(df, columns=["transaction_type", "location"], drop_first=True)

df.head()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

df.drop(columns=['timestamp'], inplace=True)  # Drop original timestamp column
df.head()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['transaction_amount'] = scaler.fit_transform(df[['transaction_amount']])
df.head()

from imblearn.over_sampling import SMOTE

X = df.drop(columns=["fraud_flag"])  # Features
y = df["fraud_flag"]  # Target

smote = SMOTE(sampling_strategy=0.3, random_state=42)  # Increase fraud cases to 30%
X_resampled, y_resampled = smote.fit_resample(X, y)

df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled["fraud_flag"] = y_resampled

df_resampled["fraud_flag"].value_counts()


# Total transactions per user
user_txn_count = df_resampled.groupby("user_id")["transaction_id"].count().rename("user_transaction_count")

# Average transaction amount per user
user_avg_amount = df_resampled.groupby("user_id")["transaction_amount"].mean().rename("user_avg_transaction_amount")

# Merge back into dataset
df_resampled = df_resampled.merge(user_txn_count, on="user_id", how="left")
df_resampled = df_resampled.merge(user_avg_amount, on="user_id", how="left")
df_resampled.head()

#Time based features
# Create a flag for night transactions (fraud is more common at night)
df_resampled["is_night"] = df_resampled["hour"].apply(lambda x: 1 if x < 6 or x > 22 else 0)

# Create a flag for weekend transactions
df_resampled["is_weekend"] = df_resampled["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

df_resampled.head()

#geographic features
# Identify all location-related columns (they start with 'location_')
location_columns = [col for col in df_resampled.columns if col.startswith("location_")]

# Count the number of unique locations per user by summing the location columns
df_resampled["user_unique_locations"] = df_resampled[location_columns].sum(axis=1)

df_resampled.head()

#Model selection & training
#Split Data into Train & Test Sets
from sklearn.model_selection import train_test_split

# Define features (X) and target variable (y)
X = df_resampled.drop(columns=["fraud_flag", "user_id", "transaction_id"])  # Remove non-predictive IDs
y = df_resampled["fraud_flag"]  # Target variable

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")

#Train logistic baseline model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Initialize and train the model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]  # Get probability scores

# Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#train random forest(better performance)
from sklearn.ensemble import RandomForestClassifier

# Initialize and train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("AUC-ROC:", roc_auc_score(y_test, y_prob_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

#train XGboost best performance
from xgboost import XGBClassifier

# Initialize and train XGBoost
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("AUC-ROC:", roc_auc_score(y_test, y_prob_xgb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))

#Trim random forest using GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define parameter grid for Random Forest
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Initialize GridSearchCV
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, 
                              scoring="roc_auc", cv=3, n_jobs=-1, verbose=2)

# Fit the model
rf_grid_search.fit(X_train, y_train)

# Best parameters & score
print("Best Parameters:", rf_grid_search.best_params_)
print("Best AUC-ROC Score:", rf_grid_search.best_score_)

# Train the best model
best_rf = rf_grid_search.best_estimator_
y_pred_rf_tuned = best_rf.predict(X_test)
y_prob_rf_tuned = best_rf.predict_proba(X_test)[:, 1]

# Evaluate performance
print("Tuned Random Forest AUC-ROC:", roc_auc_score(y_test, y_prob_rf_tuned))


import joblib
import os

# Train your model (make sure best_rf is your trained model)
# Example:
# best_rf.fit(X_train, y_train)

# Save the trained model
joblib.dump(best_rf, "fraud_detection_model.pkl")
print("Model saved as fraud_detection_model.pkl")

# Check if the file is created
if os.path.exists("fraud_detection_model.pkl"):
    print("Model file generated successfully!")


