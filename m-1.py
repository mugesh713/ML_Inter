import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.combine import SMOTEENN
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel    

# Load the dataset
file_path = 'patient_readmission_dataset.csv'
df = pd.read_csv(file_path)

# Outlier Detection and Removal using IQR method
def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
    return df
num_columns = ['LengthOfStay', 'NumberOfVisits', 'DistanceFromFacility']
df_cleaned = remove_outliers(df, num_columns)
print(df.head())
# Encode categorical features
label_encoders = {}
for column in df_cleaned.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_cleaned[column] = le.fit_transform(df_cleaned[column])
    label_encoders[column] = le

# Scaling numerical features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_cleaned[num_columns])
df_cleaned[num_columns] = scaled_features

# Define features (X) and target (y)
X = df_cleaned.drop(columns=['Readmitted'])
y = df_cleaned['Readmitted']

# Handle Class Imbalance with SMOTE-ENN
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Feature Selection
selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Hyperparameter tuning for individual models

# Random Forest Hyperparameter Tuning
rf = RandomForestClassifier(random_state=42)
rf_params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
rf_search = RandomizedSearchCV(rf, rf_params, n_iter=30, cv=5, random_state=42, n_jobs=-1)
rf_search.fit(X_train_selected, y_train)

# XGBoost Hyperparameter Tuning
xgb = XGBClassifier(random_state=42)
xgb_params = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'colsample_bytree': [0.3, 0.5, 0.7, 1.0]
}
xgb_search = RandomizedSearchCV(xgb, xgb_params, n_iter=30, cv=5, random_state=42, n_jobs=-1)
xgb_search.fit(X_train_selected, y_train)

# LightGBM Hyperparameter Tuning
lgbm = LGBMClassifier(random_state=42)
lgbm_params = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, 40, -1],  # Added -1 for unlimited depth
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [31, 40, 50, 60, 70],  # Increased range for more complexity
    'min_child_samples': [5, 10, 20]    # Adjusted for more flexibility
}
lgbm_search = RandomizedSearchCV(lgbm, lgbm_params, n_iter=30, cv=5, random_state=42, n_jobs=-1)
lgbm_search.fit(X_train_selected, y_train)

# Advanced Ensemble Learning (Stacking Classifier)
estimators = [
    ('rf', rf_search.best_estimator_),
    ('xgb', xgb_search.best_estimator_),
    ('lgbm', lgbm_search.best_estimator_)
]

stacked_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
stacked_clf.fit(X_train_selected, y_train)

# Evaluate the Ensemble Model
y_pred = stacked_clf.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Ensemble Model Accuracy:", accuracy)
print("Classification Report:\n", report)
