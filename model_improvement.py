import json
import pathlib
import pickle
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths to data
SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
EXISTING_MODEL_PATH = "model/model.pkl"
EXISTING_FEATURES_PATH = "model/model_features.json"

# Extended column selection for improved model
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode',
    # Add high-value predictors
    'waterfront', 'view', 'grade'
]

OUTPUT_DIR = "model-improved"

def load_existing_model_and_features():
    """Load the existing model created by create_model.py"""
    print("Loading existing model...")
    
    try:
        with open(EXISTING_MODEL_PATH, 'rb') as f:
            existing_model = pickle.load(f)
        
        with open(EXISTING_FEATURES_PATH, 'r') as f:
            existing_features = json.load(f)
            
        print(f"Existing model loaded successfully with {len(existing_features)} features")
        return existing_model, existing_features
    except Exception as e:
        print(f"Error loading existing model: {e}")
        return None, None

def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load data for both original and improved model evaluation."""
    print("Loading data...")
    
    # Load sales data with selected columns
    data = pd.read_csv(sales_path,
                       usecols=sales_column_selection,
                       dtype={'zipcode': str})
                       
    # Load demographics data
    demographics = pd.read_csv(demographics_path,
                              dtype={'zipcode': str})
    
    # Merge sales and demographics data
    merged_data = data.merge(demographics, how="left", on="zipcode")
    print(f"Data loaded: {merged_data.shape[0]} rows, {merged_data.shape[1]} columns")
    
    # Extract target variable
    y = merged_data.pop('price')
    X = merged_data
    
    return X, y

def find_optimal_n_neighbors(X_train_scaled, y_train):
    """Find the optimal number of neighbors using cross-validation."""
    print("\nFinding optimal number of neighbors...")
    
    best_mae = float('inf')
    best_n = 5  # Default
    
    # Test a range of neighbor values
    for n in range(3, 16, 2):  # 3, 5, 7, 9, 11, 13, 15
        # Use KFold cross-validation to get more stable results
        cv_scores = []
        kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X_train_scaled):
            # Split data
            X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Train model
            model = neighbors.KNeighborsRegressor(n_neighbors=n, weights='distance')
            model.fit(X_cv_train, y_cv_train)
            
            # Evaluate
            y_pred = model.predict(X_cv_val)
            mae = mean_absolute_error(y_cv_val, y_pred)
            cv_scores.append(mae)
        
        # Average score across folds
        avg_mae = np.mean(cv_scores)
        print(f"  n_neighbors = {n}: Average MAE = ${avg_mae:,.2f}")
        
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_n = n
    
    print(f"  Best n_neighbors: {best_n} (MAE: ${best_mae:,.2f})")
    return best_n

def evaluate_and_compare():
    """Evaluate the existing model and compare with an improved version."""
    # Load the existing model and its features
    existing_model, existing_features = load_existing_model_and_features()
    if existing_model is None:
        print("Cannot proceed without existing model.")
        return
    
    # Load data for evaluation
    X, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    
    # Create training and testing sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Evaluate existing model
    print("\n1. Evaluating existing model from create_model.py...")
    # Extract only the features expected by the existing model
    X_test_existing = X_test[existing_features].copy()
    
    # Predict using the existing model (which is a pipeline)
    y_pred_existing = existing_model.predict(X_test_existing)
    
    # Calculate metrics for existing model
    mae_existing = mean_absolute_error(y_test, y_pred_existing)
    rmse_existing = np.sqrt(mean_squared_error(y_test, y_pred_existing))
    r2_existing = r2_score(y_test, y_pred_existing)
    
    print(f"Existing model results:")
    print(f"  MAE: ${mae_existing:,.2f}")
    print(f"  RMSE: ${rmse_existing:,.2f}")
    print(f"  R²: {r2_existing:.4f}")
    
    # Create improved model
    print("\n2. Creating improved model...")
    
    # Scale features using RobustScaler
    scaler = preprocessing.RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Find optimal number of neighbors
    best_n = find_optimal_n_neighbors(X_train_scaled, y_train)
    
    # Train improved model
    improved_model = neighbors.KNeighborsRegressor(
        n_neighbors=best_n,
        weights='distance',  # Use distance weighting
        algorithm='auto'
    )
    improved_model.fit(X_train_scaled, y_train)
    
    # Evaluate improved model
    y_pred_improved = improved_model.predict(X_test_scaled)
    mae_improved = mean_absolute_error(y_test, y_pred_improved)
    rmse_improved = np.sqrt(mean_squared_error(y_test, y_pred_improved))
    r2_improved = r2_score(y_test, y_pred_improved)
    
    print(f"Improved model results:")
    print(f"  MAE: ${mae_improved:,.2f}")
    print(f"  RMSE: ${rmse_improved:,.2f}")
    print(f"  R²: {r2_improved:.4f}")
    
    # Calculate percentage improvements
    mae_pct = (mae_existing - mae_improved) / mae_existing * 100
    rmse_pct = (rmse_existing - rmse_improved) / rmse_existing * 100
    r2_pct = (r2_improved - r2_existing) / abs(r2_existing) * 100
    
    print("\n3. Improvement summary:")
    print(f"  MAE: {'improved' if mae_pct > 0 else 'worsened'} by {abs(mae_pct):.2f}%")
    print(f"  RMSE: {'improved' if rmse_pct > 0 else 'worsened'} by {abs(rmse_pct):.2f}%")
    print(f"  R²: {'improved' if r2_pct > 0 else 'worsened'} by {abs(r2_pct):.2f}%")
    
    # Save improved model and features
    print("\n4. Saving improved model...")
    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Create separate directory for improved model to avoid overwriting
    improved_dir = output_dir / "improved"
    improved_dir.mkdir(exist_ok=True)
    
    # Save improved model
    with open(improved_dir / "model.pkl", 'wb') as f:
        pickle.dump(improved_model, f)
    
    # Save feature list
    with open(improved_dir / "model_features.json", 'w') as f:
        json.dump(list(X_train.columns), f)
    
    # Save scaler
    with open(improved_dir / "model_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"Improved model saved to {improved_dir}/model.pkl")
    print(f"Features saved to {improved_dir}/model_features.json")
    print(f"Scaler saved to {improved_dir}/model_scaler.pkl")
    
    return {
        'existing': {
            'mae': mae_existing,
            'rmse': rmse_existing,
            'r2': r2_existing
        },
        'improved': {
            'mae': mae_improved,
            'rmse': rmse_improved,
            'r2': r2_improved
        }
    }


if __name__ == "__main__":
    evaluate_and_compare()
