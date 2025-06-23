import numpy as np
import matplotlib.pyplot as plt

class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return {'value': np.mean(y), 'samples': y.tolist()}

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return {'value': np.mean(y), 'samples': y.tolist()}

        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold

        if sum(left_indices) == 0 or sum(right_indices) == 0:
            return {'value': np.mean(y), 'samples': y.tolist()}

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature': feature, 'threshold': threshold, 'left': left_tree, 'right': right_tree}

    def _best_split(self, X, y):
        best_feature, best_threshold, best_loss = None, None, float('inf')

        for feature in range(X.shape[1]):
            thresholds, losses = self._split_loss(X[:, feature], y)
            if losses.size == 0:
                continue
            if losses.min() < best_loss:
                best_loss = losses.min()
                best_threshold = thresholds[losses.argmin()]
                best_feature = feature

        return best_feature, best_threshold

    def _split_loss(self, feature, y):
        thresholds = np.unique(feature)
        losses = np.array([self._loss(y, feature < threshold) for threshold in thresholds])
        return thresholds, losses

    def _loss(self, y, split):
        if sum(split) == 0 or sum(~split) == 0:
            return float('inf')
        left_loss = np.mean((y[split] - y[split].mean()) ** 2)
        right_loss = np.mean((y[~split] - y[~split].mean()) ** 2)
        return len(y[split]) * left_loss + len(y[~split]) * right_loss

    def _predict_one(self, x, tree):
        if 'feature' not in tree:
            return tree['value']
        feature, threshold = tree['feature'], tree['threshold']
        if x[feature] < threshold:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])

    def print_tree(self, tree=None, depth=0):
        if tree is None:
            tree = self.tree

        if 'feature' not in tree:
            print("\t" * depth + f"Leaf: value = {tree['value']:.4f}, samples = {tree['samples']}")
        else:
            print("\t" * depth + f"[X{tree['feature']} < {tree['threshold']:.4f}]")
            self.print_tree(tree['left'], depth + 1)
            print("\t" * depth + f"[X{tree['feature']} >= {tree['threshold']:.4f}]")
            self.print_tree(tree['right'], depth + 1)

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.initial_prediction = None
        self.trees = []

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        self.predictions = np.full(y.shape, self.initial_prediction)

        for _ in range(self.n_estimators):
            residuals = y - self.predictions
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            self.predictions += self.learning_rate * predictions
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

# Enhanced example usage with a small dataset having two features
if __name__ == "__main__":
    # Define a small dataset with 10 points and 2 features
    np.random.seed(42)
    X = np.random.rand(10, 2) * 10
    y = X[:, 0] * 0.5 + X[:, 1] * 2.0 + np.random.randn(10) * 0.1  # y depends on both features

    # Train the model
    model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=3)
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Print predictions
    for true_val, pred_val in zip(y, y_pred):
        print(f"True value: {true_val:.2f}, Predicted value: {pred_val:.2f}")

    # Print the structure of the first tree in the model
    print("\nTree Structure:")
    model.trees[0].print_tree()

    # Scatter plot for the dataset
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=100, edgecolor='k')
    plt.colorbar(label='Target Value')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter Plot of the Dataset with Two Features')
    plt.grid(True)
    plt.show()
