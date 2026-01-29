import numpy as np
import pandas as pd
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

class GraphFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Constructs a k-NN graph on the training data and extracts graph-theoretic features.
    For new data (inductive step), aggregates features from nearest neighbors in the training graph.
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.nn_model = None
        self.train_graph_features = None
        
    def fit(self, X, y=None):
        # Ensure X is numpy array for fitting
        X_val = X.values if isinstance(X, pd.DataFrame) else X
        
        # 1. Fit Nearest Neighbors on the training set
        self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.nn_model.fit(X_val)
        
        # 2. Build k-NN Graph (k neighbors for each node)
        # mode='connectivity' returns 1s and 0s
        adj = kneighbors_graph(self.nn_model, self.n_neighbors, mode='connectivity', include_self=False)
        G = nx.from_scipy_sparse_array(adj)
        
        # 3. Compute Graph Features (Centrality, Clustering, etc.)
        # PageRank (Measure of node importance)
        pr = nx.pagerank(G, alpha=0.85)
        # Degree Centrality (Measure of connections)
        deg = nx.degree_centrality(G)
        # Clustering Coefficient (Measure of local clique structure)
        cc = nx.clustering(G)
        
        # 4. Store features in an array aligned with X indices (0 to N-1)
        n_samples = X_val.shape[0]
        self.train_graph_features = np.zeros((n_samples, 3))
        
        for i in range(n_samples):
            self.train_graph_features[i, 0] = pr.get(i, 0)
            self.train_graph_features[i, 1] = deg.get(i, 0)
            self.train_graph_features[i, 2] = cc.get(i, 0)
             
        return self

    def transform(self, X):
        X_val = X.values if isinstance(X, pd.DataFrame) else X
        
        if self.nn_model is None or self.train_graph_features is None:
            raise RuntimeError("GraphFeatureExtractor must be fit before transform.")

        # For transform, we find neighbors in the training set for each new sample
        # and average their graph features (Label/Feature Propagation).
        dists, indices = self.nn_model.kneighbors(X_val)
        
        # Fetch features of neighbors: shape (n_queries, n_neighbors, 3)
        neighbor_features = self.train_graph_features[indices] 
        
        # Average aggregation
        new_feats = np.mean(neighbor_features, axis=1)
        
        if isinstance(X, pd.DataFrame):
             feat_names = ['pagerank_mean', 'degree_mean', 'clustering_mean']
             new_df = pd.DataFrame(new_feats, columns=feat_names, index=X.index)
             return pd.concat([X, new_df], axis=1)
        else:
             return np.hstack([X_val, new_feats])
