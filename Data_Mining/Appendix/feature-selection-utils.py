import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
from scipy import stats

class FeatureSelector:
    """A utility class for performing various types of feature selection."""
    
    def __init__(self, X, y):
        """
        Initialize the FeatureSelector with data.
        
        Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        """
        self.X = X
        self.y = y
        self.feature_names = X.columns
        
    def correlation_filter(self, threshold=0.8):
        """
        Remove highly correlated features.
        
        Parameters:
        threshold (float): Correlation coefficient threshold (default: 0.7)
        
        Returns:
        list: Selected feature names
        """
        corr_matrix = self.X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_features = [column for column in upper.columns if any(upper[column] > threshold)]
        selected_features = [col for col in self.feature_names if col not in drop_features]
        
        return selected_features
    
    def variance_filter(self, threshold=0.01):
        """
        Remove low-variance features.
        
        Parameters:
        threshold (float): Variance threshold (default: 0.01)
        
        Returns:
        list: Selected feature names
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        variances = np.var(X_scaled, axis=0)
        selected_features = self.feature_names[variances > threshold]
        
        return selected_features.tolist()
    
 
    def decision_tree_importance(self, k=10):
        """
        Select top k features based on decision tree importance.
        
        Parameters:
        k (int): Number of features to select (default: 10)
        
        Returns:
        list: Selected feature names
        """
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(self.X, self.y)
        
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': dt.feature_importances_
        })
        selected_features = importances.nlargest(k, 'importance')['feature'].tolist()
        
        return selected_features
    
    def lasso_selection(self, alpha=1.0):
        """
        Select features using Lasso regularization.
        
        Parameters:
        alpha (float): Regularization strength (default: 1.0)
        
        Returns:
        list: Selected feature names
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Fit Lasso
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_scaled, self.y)
        
        # Get features with non-zero coefficients
        selected_features = self.feature_names[np.abs(lasso.coef_) > 1e-3].tolist()
        
        return selected_features


from sklearn.datasets import make_classification

# Create a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, n_redundant=5, random_state=42)

# Convert to DataFrame
X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
y_series = pd.Series(y)

# Initialize the FeatureSelector
selector = FeatureSelector(X_df, y_series)

# Try each method individually
print("1. Correlation Filter:")
print(selector.correlation_filter())
print("\n2. Variance Filter:")
print(selector.variance_filter())
print("\n4. Decision Tree Importance (top 5 features):")
print(selector.decision_tree_importance(k=5))
print("\n5. Lasso Selection:")
print(selector.lasso_selection(alpha=0.1))
