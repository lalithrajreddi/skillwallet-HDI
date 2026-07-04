import pandas as pd
import numpy as np
import urllib.request
import os

def download_and_preprocess():
    print("Starting dataset download and preprocessing...")
    
    url = "https://raw.githubusercontent.com/yhpong/Scientific-Toolkit/master/HDI_Data.csv"
    local_raw_path = "hdi_raw.csv"
    local_processed_path = "hdi_processed.csv"
    
    # Step 1: Download raw dataset if not present
    if not os.path.exists(local_raw_path):
        print(f"Downloading raw dataset from {url}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(local_raw_path, 'wb') as out_file:
                out_file.write(response.read())
            print("Download completed successfully.")
        except Exception as e:
            print(f"Error downloading data: {e}")
            return False
    else:
        print("Raw dataset already exists locally.")

    # Step 2: Load dataset
    df = pd.read_csv(local_raw_path, encoding='latin1', keep_default_na=False)
    print("Initial dataset shape:", df.shape)
    print("Initial columns:", df.columns.tolist())
    
    # Step 3: Clean columns (convert string numeric values to float, handle commas)
    numeric_cols = ['Life expectancy', 'Expected yrs of schooling', 'Mean yrs of schooling', 'GNI per capita']
    
    for col in numeric_cols:
        if col in df.columns:
            # Replace commas and spaces, coerce to float
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Step 4: Handle missing values (fill null values using column mean methods)
    print("\nNull values before filling:")
    print(df[numeric_cols].isnull().sum())
    
    for col in numeric_cols:
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
        print(f"Filled nulls in '{col}' with mean: {mean_val:.2f}")
        
    print("\nNull values after filling:")
    print(df[numeric_cols].isnull().sum())
    
    # Step 5: Perform Feature Engineering (Calculate indices and composite HDI Score)
    print("\nCalculating indices and composite HDI Score...")
    
    # A: Health Index = (Life expectancy - 20) / (85 - 20)
    df['Health_Index'] = (df['Life expectancy'] - 20) / (85 - 20)
    df['Health_Index'] = df['Health_Index'].clip(0, 1) # clip to [0, 1]
    
    # B: Education Index = (Expected yrs of schooling index + Mean yrs of schooling index) / 2
    # Expected schooling index = Expected yrs of schooling / 18
    df['Expected_Schooling_Index'] = df['Expected yrs of schooling'] / 18
    df['Expected_Schooling_Index'] = df['Expected_Schooling_Index'].clip(0, 1)
    
    # Mean schooling index = Mean yrs of schooling / 15
    df['Mean_Schooling_Index'] = df['Mean yrs of schooling'] / 15
    df['Mean_Schooling_Index'] = df['Mean_Schooling_Index'].clip(0, 1)
    
    df['Education_Index'] = (df['Expected_Schooling_Index'] + df['Mean_Schooling_Index']) / 2
    
    # C: Income Index = (ln(GNI per capita) - ln(100)) / (ln(75000) - ln(100))
    # GNI per capita must be > 0. Standard UNDP caps GNI at 75000 and 100.
    df['GNI_Cleaned'] = df['GNI per capita'].clip(100, 75000)
    df['Income_Index'] = (np.log(df['GNI_Cleaned']) - np.log(100)) / (np.log(75000) - np.log(100))
    df['Income_Index'] = df['Income_Index'].clip(0, 1)
    
    # D: Calculate geometric mean of the three dimensions for HDI Score
    df['HDI Score'] = (df['Health_Index'] * df['Education_Index'] * df['Income_Index']) ** (1/3)
    df['HDI Score'] = df['HDI Score'].round(3)
    
    # E: Classify country into development tier
    # Very High: >= 0.800
    # High: 0.700 - 0.799
    # Medium: 0.550 - 0.699
    # Low: < 0.550
    def categorize_hdi(score):
        if score >= 0.800:
            return 'Very High'
        elif score >= 0.700:
            return 'High'
        elif score >= 0.550:
            return 'Medium'
        else:
            return 'Low'
            
    df['Development Tier'] = df['HDI Score'].apply(categorize_hdi)
    
    # Save the processed dataset
    df.to_csv(local_processed_path, index=False)
    print(f"\nSaved processed dataset to '{local_processed_path}' (Shape: {df.shape})")
    print(df[['Country Name', 'Life expectancy', 'GNI per capita', 'HDI Score', 'Development Tier']].head())
    return True

if __name__ == "__main__":
    download_and_preprocess()
