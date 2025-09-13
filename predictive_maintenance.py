import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load the datasets
vehicle_info_df = pd.read_csv('sample_data/vehicle_info.csv')
maintenance_logs_df = pd.read_csv('sample_data/maintenance_logs.csv')
sensor_data_df = pd.read_csv('sample_data/sensor_data.csv')
driver_behavior_df = pd.read_csv('sample_data/driver_behavior.csv')

# --- Data Preprocessing and Feature Engineering ---

# Convert date columns to datetime objects
sensor_data_df['timestamp'] = pd.to_datetime(sensor_data_df['timestamp'])
maintenance_logs_df['maintenance_date'] = pd.to_datetime(maintenance_logs_df['maintenance_date'])
driver_behavior_df['timestamp'] = pd.to_datetime(driver_behavior_df['timestamp'])

# Merge the datasets
merged_df = pd.merge(sensor_data_df, vehicle_info_df, on='truck_id', how='left')
merged_df = pd.merge(merged_df, driver_behavior_df, on=['truck_id', 'timestamp'], how='left')

# Create Target Variable
failure_events = maintenance_logs_df[maintenance_logs_df['reason'] == 'Failure']
merged_df['failure_imminent'] = 0

for index, row in failure_events.iterrows():
    truck_id = row['truck_id']
    failure_date = row['maintenance_date']
    start_date = failure_date - pd.Timedelta(days=30)
    
    merged_df.loc[
        (merged_df['truck_id'] == truck_id) & 
        (merged_df['timestamp'] >= start_date) & 
        (merged_df['timestamp'] < failure_date), 
        'failure_imminent'] = 1

# Feature Engineering
merged_df = merged_df.sort_values(by=['truck_id', 'timestamp'])

sensor_cols = ['engine_temp', 'oil_pressure', 'tire_pressure', 'vibration']
for col in sensor_cols:
    merged_df[f'{col}_rolling_avg'] = merged_df.groupby('truck_id')[col].transform(lambda x: x.rolling(window=10, min_periods=1).mean())

driver_behavior_dummies = pd.get_dummies(merged_df['event'], prefix='event')
merged_df = pd.concat([merged_df, driver_behavior_dummies], axis=1)

# Data Cleaning
merged_df.drop(['reading_id', 'event', 'behavior_id'], axis=1, inplace=True)
merged_df.fillna(0, inplace=True)

# One-hot encoding
final_df = pd.get_dummies(merged_df, columns=['model'], drop_first=True)

# --- Model Training and Evaluation ---

X = final_df.drop(['truck_id', 'timestamp', 'failure_imminent'], axis=1)
y = final_df['failure_imminent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Logistic Regression ---
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)

print("Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_log_reg):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_log_reg):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_log_reg):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_log_reg):.4f}")
sns.heatmap(confusion_matrix(y_test, y_pred_log_reg), annot=True, fmt='d')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# --- Random Forest ---
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train_scaled, y_train)
y_pred_rf = rf_clf.predict(X_test_scaled)

print("Random Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_rf):.4f}")
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d')
plt.title('Random Forest Confusion Matrix')
plt.show()

# --- XGBoost ---
xgb_clf = XGBClassifier(random_state=42)
xgb_clf.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_clf.predict(X_test_scaled)

print("XGBoost Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_xgb):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred_xgb):.4f}")
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d')
plt.title('XGBoost Confusion Matrix')
plt.show()

# --- Prediction ---

# Load the prediction set
predict_df = pd.read_csv('sample_data/predict_set.csv')

# Preprocess the prediction set
predict_df['timestamp'] = pd.to_datetime(predict_df['timestamp'])
predict_df = pd.merge(predict_df, vehicle_info_df, on='truck_id', how='left')

# Feature Engineering for prediction set
predict_df = predict_df.sort_values(by=['truck_id', 'timestamp'])

for col in sensor_cols:
    predict_df[f'{col}_rolling_avg'] = predict_df.groupby('truck_id')[col].transform(lambda x: x.rolling(window=10, min_periods=1).mean())

driver_behavior_dummies_pred = pd.get_dummies(predict_df['event'], prefix='event')
predict_df = pd.concat([predict_df, driver_behavior_dummies_pred], axis=1)

# Data Cleaning for prediction set
predict_df.drop(['event'], axis=1, inplace=True)
predict_df.fillna(0, inplace=True)

# One-hot encoding for prediction set
predict_df = pd.get_dummies(predict_df, columns=['model'], drop_first=True)

# Align columns with the training data
predict_X = predict_df[X.columns]

# Scale the prediction set
predict_X_scaled = scaler.transform(predict_X)

# Make predictions
predictions = xgb_clf.predict(predict_X_scaled)
prediction_probas = xgb_clf.predict_proba(predict_X_scaled)

# Display the predictions
predict_df['failure_prediction'] = predictions
predict_df['failure_probability'] = prediction_probas[:, 1]

print("Predictions on the prediction set:")
display(predict_df[['truck_id', 'timestamp', 'failure_prediction', 'failure_probability']])
