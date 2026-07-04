import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import os

def train_hdi_model():
    print("Starting Machine Learning Model Training...")
    local_processed_path = "hdi_processed.csv"
    model_path = "hdi_model.pkl"
    
    if not os.path.exists(local_processed_path):
        print(f"Error: Processed dataset '{local_processed_path}' not found! Run download_data.py first.")
        return False
        
    # Step 1: Load preprocessed dataset
    df = pd.read_csv(local_processed_path)
    
    # Step 2: Select independent (features) and dependent (target) variables
    X = df[['Life expectancy', 'Expected yrs of schooling', 'Mean yrs of schooling', 'GNI per capita']]
    y = df['HDI Score']
    
    # Step 3: Split dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset split: Train shape = {X_train.shape}, Test shape = {X_test.shape}")
    
    # Step 4: Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model successfully trained.")
    
    # Print model parameters
    print("\nModel Coefficients:")
    for col, coef in zip(X.columns, model.coef_):
        print(f"  {col}: {coef:.6f}")
    print(f"  Intercept: {model.intercept_:.6f}")
    
    # Step 5: Evaluate model performance
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    print("\nTraining Set Metrics:")
    print(f"  R-Squared (R2): {metrics.r2_score(y_train, train_pred):.4f}")
    print(f"  Mean Absolute Error (MAE): {metrics.mean_absolute_error(y_train, train_pred):.4f}")
    print(f"  Mean Squared Error (MSE): {metrics.mean_squared_error(y_train, train_pred):.4f}")
    
    print("\nTesting Set Metrics (Generalization):")
    r2 = metrics.r2_score(y_test, test_pred)
    mae = metrics.mean_absolute_error(y_test, test_pred)
    mse = metrics.mean_squared_error(y_test, test_pred)
    rmse = np.sqrt(mse)
    
    print(f"  R-Squared (R2): {r2:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    # Step 6: Save model using Pickle serialization
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nSerialized model saved to '{model_path}' using Pickle.")
    
    # Save a small validation info file to log metrics
    with open("model_summary.txt", "w") as f:
        f.write("HDI PREDICTOR MODEL SUMMARY\n")
        f.write("===========================\n")
        f.write(f"Algorithm: Linear Regression\n")
        f.write(f"Features: {X.columns.tolist()}\n")
        f.write(f"Train/Test split: 80/20\n\n")
        f.write(f"Testing Performance:\n")
        f.write(f"  R2 Score: {r2:.6f}\n")
        f.write(f"  MAE: {mae:.6f}\n")
        f.write(f"  MSE: {mse:.6f}\n")
        f.write(f"  RMSE: {rmse:.6f}\n\n")
        f.write("Coefficients:\n")
        for col, coef in zip(X.columns, model.coef_):
            f.write(f"  {col}: {coef:.6f}\n")
        f.write(f"  Intercept: {model.intercept_:.6f}\n")
        
    return True

if __name__ == "__main__":
    train_hdi_model()
