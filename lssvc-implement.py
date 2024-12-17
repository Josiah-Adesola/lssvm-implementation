import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler  # Changed from MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from lssvm import LSSVC
import warnings
from typing import Tuple, Dict

def find_best_parameters(X_train: np.ndarray, y_train: np.ndarray, kernel: str) -> Dict:
    """
    Find optimal parameters using cross-validation.
    """
    param_grids = {
        'rbf': {
            'gamma': [0.001, 0.01, 0.1, 1.0],
            'sigma': [0.1, 0.5, 1.0, 2.0]
        },
        'linear': {
            'gamma': [0.001, 0.01, 0.1, 1.0]
        },
        'poly': {
            'gamma': [0.001, 0.01, 0.1],
            'd': [2, 3]
        }
    }
    
    best_params = {}
    best_score = 0
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    if kernel == 'rbf':
        for gamma in param_grids['rbf']['gamma']:
            for sigma in param_grids['rbf']['sigma']:
                scores = []
                for train_idx, val_idx in cv.split(X_train):
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                    
                    model = LSSVC(gamma=gamma, kernel='rbf', sigma=sigma)
                    model.fit(X_fold_train, y_fold_train)
                    y_pred = model.predict(X_fold_val)
                    scores.append(accuracy_score(y_fold_val, y_pred))
                
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {'gamma': gamma, 'sigma': sigma}
    
    elif kernel == 'linear':
        for gamma in param_grids['linear']['gamma']:
            scores = []
            for train_idx, val_idx in cv.split(X_train):
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                model = LSSVC(gamma=gamma, kernel='linear')
                model.fit(X_fold_train, y_fold_train)
                y_pred = model.predict(X_fold_val)
                scores.append(accuracy_score(y_fold_val, y_pred))
            
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = {'gamma': gamma}
    
    print(f"Best cross-validation score: {best_score:.3f}")
    return best_params

def preprocess_data(X_train, X_test, y_train, y_test):
    """
    Preprocess the data with improved scaling and normalization.
    """
    # Use StandardScaler instead of MinMaxScaler for better numerical stability
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ensure labels are in the correct format
    lb = LabelBinarizer()
    if len(np.unique(y_train)) == 2:
        y_train_processed = lb.fit_transform(y_train).ravel()
        y_test_processed = lb.transform(y_test).ravel()
    else:
        y_train_processed = y_train
        y_test_processed = y_test
    
    return X_train_scaled, X_test_scaled, y_train_processed, y_test_processed

def main():
    # Load your data
    print("Loading and preprocessing data...")
    # Replace these with your actual data loading
    X, y = load_digits(return_X_y=True)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocess the data
    X_train_scaled, X_test_scaled, y_train_processed, y_test_processed = preprocess_data(
        X_train, X_test, y_train, y_test
    )
    
    # Try different kernels
    kernels = ['rbf', 'linear']
    best_accuracy = 0
    best_kernel = None
    best_model = None
    
    for kernel in kernels:
        print(f"\nTraining with {kernel} kernel:")
        
        # Find best parameters for the current kernel
        best_params = find_best_parameters(X_train_scaled, y_train_processed, kernel)
        print(f"Best parameters: {best_params}")
        
        # Train model with best parameters
        model = LSSVC(kernel=kernel, **best_params)
        model.fit(X_train_scaled, y_train_processed)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_processed, y_pred)
        print(f"Test accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test_processed, y_pred))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_kernel = kernel
            best_model = model
    
    print(f"\nBest Results:")
    print(f"Kernel: {best_kernel}")
    print(f"Accuracy: {best_accuracy:.3f}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
