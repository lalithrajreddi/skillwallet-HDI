import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import os

def train_hdi_model():
    print("Starting Machine Learning Model Training (Epic 4 & 5 Preprocessing in Training stage)...")
    local_raw_path = "hdi_raw.csv"
    local_processed_path = "hdi_processed.csv"
    model_path = "hdi_model.pkl"
    encoder_path = "country_encoder.pkl"
    
    if not os.path.exists(local_raw_path):
        print(f"Error: Raw dataset '{local_raw_path}' not found! Run download_data.py first.")
        return False
        
    # Step 1: Load raw dataset (195 rows x 82 columns)
    df = pd.read_csv(local_raw_path)
    print("Loaded raw dataset shape:", df.shape)
    
    # Step 2: Select independent (features X) and dependent (target y) variables by column index numbers
    # Spec requirements:
    # Index 2: Country Name (categorical) -> Stored at column index 0 in X
    # Index 4: HDI Score (Y prediction target)
    # Index 5: Life expectancy
    # Index 6: Expected yrs of schooling
    # Index 7: Mean yrs of schooling
    # Index 8: GNI per capita
    X_raw = df.iloc[:, [2, 5, 6, 7, 8]].copy()
    y = df.iloc[:, 4]
    
    print("Selected raw features columns in X_raw:", X_raw.columns.tolist())
    print("Selected target column y:", y.name)
    
    # Step 3: Check for missing values (X.isnull().sum())
    print("\nChecking null values in X_raw before cleaning:")
    print(X_raw.isnull().sum())
    
    # Step 4: Fill Null Values in X using column mean for numeric columns
    numeric_cols = X_raw.columns[1:] # indices 5, 6, 7, 8 in main df
    for col in numeric_cols:
        mean_val = X_raw[col].mean()
        X_raw[col] = X_raw[col].fillna(mean_val)
        
    # Step 5: Perform Label Encoding on Country Name
    print("\nPerforming Label Encoding on Country Name...")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # Fill null country name with "Unknown" if any
    X_raw.iloc[:, 0] = X_raw.iloc[:, 0].fillna("Unknown")
    X_raw.iloc[:, 0] = le.fit_transform(X_raw.iloc[:, 0])
    
    # Save the label encoder for Flask backend
    with open(encoder_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"Saved LabelEncoder to '{encoder_path}'")
    
    X = X_raw
    
    # Step 6: Split dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset split: Train shape = {X_train.shape}, Test shape = {X_test.shape}")
    
    # Step 7: Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression model successfully trained.")
    
    # Print model parameters
    print("\nModel Coefficients:")
    for col, coef in zip(X.columns, model.coef_):
        print(f"  {col}: {coef:.6f}")
    print(f"  Intercept: {model.intercept_:.6f}")
    
    # Step 8: Evaluate model performance
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
    
    # Step 9: Save model using Pickle serialization
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nSerialized model saved to '{model_path}' using Pickle.")
    
    # Save a validation summary log
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
        
    # Step 10: Reconstruct processed file for the Flask Web Application
    processed_df = pd.DataFrame()
    processed_df['Rank'] = range(1, len(df) + 1)
    processed_df['Country Name'] = df['Country Name']
    processed_df['Country_Encoded'] = X.iloc[:, 0]
    processed_df['Country ISO'] = df['Country Name'].str[:3].str.upper()
    processed_df['HDI Score'] = y
    
    # Recover padded column features for indices
    processed_df['Life expectancy'] = df['Life expectancy'].fillna(df['Life expectancy'].mean())
    processed_df['Expected yrs of schooling'] = df['Expected yrs of schooling'].fillna(df['Expected yrs of schooling'].mean())
    processed_df['Mean yrs of schooling'] = df['Mean yrs of schooling'].fillna(df['Mean yrs of schooling'].mean())
    processed_df['GNI per capita'] = df['GNI per capita'].fillna(df['GNI per capita'].mean())
    
    # Recompute tiers
    def categorize_hdi(score):
        if score >= 0.800:
            return 'Very High'
        elif score >= 0.700:
            return 'High'
        elif score >= 0.550:
            return 'Medium'
        else:
            return 'Low'
    processed_df['Development Tier'] = processed_df['HDI Score'].apply(categorize_hdi)
    
    # Compute indices
    processed_df['Health_Index'] = ((processed_df['Life expectancy'] - 20) / 65.0).clip(0, 1)
    processed_df['Expected_Schooling_Index'] = (processed_df['Expected yrs of schooling'] / 18.0).clip(0, 1)
    processed_df['Mean_Schooling_Index'] = (processed_df['Mean yrs of schooling'] / 15.0).clip(0, 1)
    processed_df['Education_Index'] = (processed_df['Expected_Schooling_Index'] + processed_df['Mean_Schooling_Index']) / 2.0
    
    gni_cleaned = processed_df['GNI per capita'].clip(100, 75000)
    processed_df['GNI_Cleaned'] = gni_cleaned
    processed_df['Income_Index'] = ((np.log(gni_cleaned) - np.log(100.0)) / (np.log(75000.0) - np.log(100.0))).clip(0, 1)
    
    processed_df.to_csv(local_processed_path, index=False)
    print(f"SUCCESS! Formatted processed dataset saved to '{local_processed_path}' (Shape: {processed_df.shape})")
    
    return True

if __name__ == "__main__":
    train_hdi_model()
