from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from .graph_features import GraphFeatureExtractor

def build_pipeline(classifier_type='rf', n_neighbors=10):
    """
    Builds the ML pipeline with Graph Feature Engineering.
    
    Args:
        classifier_type (str): 'rf' for RandomForest, 'xgb' for XGBoost.
        n_neighbors (int): Number of neighbors for graph construction.
    """
    # 1. Scaling is important for k-NN graph construction
    scaler = StandardScaler()
    
    # 2. Graph Feature Extractor
    graph_fe = GraphFeatureExtractor(n_neighbors=n_neighbors)
    
    # 3. High-performance Classifier
    if classifier_type == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier_type == 'xgb':
        # use_label_encoder=False is deprecated in newer versions but safe to omit usually
        clf = XGBClassifier(eval_metric='logloss', random_state=42)
    else:
        raise ValueError("Unknown classifier type")
        
    pipeline = Pipeline([
        ('scaler', scaler),
        ('graph_features', graph_fe), # Adds columns to X
        ('classifier', clf)
    ])
    
    return pipeline

def evaluate_model(pipeline, X, y, cv=5):
    """
    Performs Cross-Validation using Stratified K-Fold.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy')
    return scores
