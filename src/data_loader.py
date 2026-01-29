import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_data():
    """
    Loads the Breast Cancer Wisconsin dataset.
    
    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Target labels (0 = malignant, 1 = benign).
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y

if __name__ == "__main__":
    X, y = load_data()
    print(f"Data loaded successfully. Shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
