import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def train_model(data_path, model_path):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Select Features
    # Note: We use 'LeadTime' and 'Age' (numeric) + others (categorical)
    numeric_features = ['Age', 'LeadTime', 'ChronicConditions']
    categorical_features = ['Gender', 'Neighbourhood', 'Scholarship', 'SMS_received', 'DayOfWeek']
    
    X = df[numeric_features + categorical_features]
    y = df['NoShowBinary']

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Model Pipeline (Logistic Regression is standard for baseline)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])

    # Split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    print("Training Logistic Regression...")
    model.fit(X_train, y_train)

    # Evaluation
    print("Evaluating...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save Model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save Test Data for Evaluation Script
    test_data = X_test.copy()
    test_data['NoShowBinary'] = y_test
    test_data.to_csv(os.path.join(os.path.dirname(data_path), "test_data.csv"), index=False)
    print(f"Test data saved to {os.path.join(os.path.dirname(data_path), 'test_data.csv')}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "no_show_features.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "log_reg_model.pkl")

    if os.path.exists(INPUT_PATH):
        train_model(INPUT_PATH, MODEL_PATH)
    else:
        print("Error: Feature data not found. Run feature_engineering.py first.")
