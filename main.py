import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure src can be imported
sys.path.append('.')

from src.data_loader import load_data
from src.model import build_pipeline, evaluate_model

def main():
    print("Loading data...")
    X, y = load_data()
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n--- Model Development ---")
    print("Building pipeline with XGBoost and Dataset-Induced Graph Features...")
    pipeline = build_pipeline(classifier_type='xgb', n_neighbors=10)
    
    print("Performing 5-fold Cross-Validation on Training Set...")
    cv_scores = evaluate_model(pipeline, X_train, y_train, cv=5)
    print(f"CV Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    print("\n--- Model Training & Verification ---")
    print("Training final model on full training set...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating on Test Set...")
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix\nTest Acc: {test_acc:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    output_img = 'confusion_matrix.png'
    plt.savefig(output_img)
    print(f"Confusion matrix saved to {output_img}")

if __name__ == "__main__":
    main()
