import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

def evaluate_model(model_path, test_data_path, output_dir):
    print("Loading resources...")
    model = joblib.load(model_path)
    df_test = pd.read_csv(test_data_path)
    
    X_test = df_test.drop(columns=['NoShowBinary'])
    y_test = df_test['NoShowBinary']
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    print(f"Evaluation plots saved to {output_dir}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "log_reg_model.pkl")
    TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "test_data.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "figures")
    
    if os.path.exists(MODEL_PATH) and os.path.exists(TEST_DATA_PATH):
        evaluate_model(MODEL_PATH, TEST_DATA_PATH, OUTPUT_DIR)
    else:
        print("Model or Test Data not found. Run model_training.py first.")
