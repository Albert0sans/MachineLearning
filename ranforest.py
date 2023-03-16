##https://link.springer.com/content/pdf/10.1023%2FA%3A1010933404324.pdf
# TODO: make a class for the random forest algorithm not yet WORking !!!!
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_size=1, sample_size=1.0, n_features=None):
        self.trees = []
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.n_features = n_features
        for _ in range(self.n_trees):
            self.trees.append(DecisionTree(max_depth=self.max_depth, min_size=self.min_size, sample_size=self.sample_size, n_features=self.n_features))
    
    def fit(self, X, y):
        for tree in self.trees:
            tree.fit(X, y)
    
    def predict(self, X):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
        return np.array(predictions).mean(axis=0)
    
class DecisionTree:
    def __init__(self, max_depth=5, min_size=1, sample_size=1.0, n_features=None):
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.n_features = n_features
        self.root = None
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
    
    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_size):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Grow tree
        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)
        print(feature_idxs)
        
        # Greedily select best split according to information gain
        best_idx, best_threshold = self._best_criteria(X, y, feature_idxs)
        
        # Build subtrees
        left_idxs, right_idxs = self._split(X[:, best_idx], best_threshold)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_idx, best_threshold, left, right)
    
    def _best_criteria(self, X, y, feature_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        for idx in feature_idxs:
            X_column = X[:, idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_threshold = threshold
                    
        return split_idx, split_threshold
    
    def _information_gain(self, y, X_column, split_threshold):
        # Parent loss
        parent_loss = self._gini(y)
        
        # Generate split
        left_idxs, right_idxs = self._split(X_column, split_threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
class Node:
    def __init__(self, index=None, threshold=None, left=None, right=None, *, value=None):
        self.index = index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def _predict(self, inputs):
        # Check if leaf node
        if self.value is not None:
            return self.value
        
        if inputs[self.index] < self.threshold:
            return self.left._predict(inputs)
        return self.right._predict(inputs)
    
    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities ** 2)
    
    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
    
    def _split(self, X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs
    
    
if __name__ == "__main__":
    ## test the random forest algorithm

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    clf = RandomForest(n_trees=3, max_depth=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    accuracy = accuracy_score(y_test, y_pred)
