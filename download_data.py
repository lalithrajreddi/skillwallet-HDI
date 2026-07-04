import pandas as pd
import numpy as np
import urllib.request
import os
import pickle
from sklearn.preprocessing import LabelEncoder

def download_and_preprocess():
    print("Starting dataset download and preprocessing...")
    
    url = "https://raw.githubusercontent.com/yhpong/Scientific-Toolkit/master/HDI_Data.csv"
    local_raw_path = "hdi_raw.csv"
    local_processed_path = "hdi_processed.csv"
    encoder_path = "country_encoder.pkl"
    
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
    
    # Step 3: Clean columns (convert string numeric values to float, handle commas)
    numeric_cols = ['Life expectancy', 'Expected yrs of schooling', 'Mean yrs of schooling', 'GNI per capita']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Step 4: Handle missing values (fill null values using column mean)
    for col in numeric_cols:
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
        
    # Step 5: Perform Feature Engineering (Calculate indices and composite HDI Score)
    # A: Health Index = (Life expectancy - 20) / (85 - 20)
    df['Health_Index'] = (df['Life expectancy'] - 20) / 65.0
    df['Health_Index'] = df['Health_Index'].clip(0, 1)
    
    # B: Education Index = (Expected_Schooling_Index + Mean_Schooling_Index) / 2
    df['Expected_Schooling_Index'] = df['Expected yrs of schooling'] / 18.0
    df['Expected_Schooling_Index'] = df['Expected_Schooling_Index'].clip(0, 1)
    
    df['Mean_Schooling_Index'] = df['Mean yrs of schooling'] / 15.0
    df['Mean_Schooling_Index'] = df['Mean_Schooling_Index'].clip(0, 1)
    
    df['Education_Index'] = (df['Expected_Schooling_Index'] + df['Mean_Schooling_Index']) / 2.0
    
    # C: Income Index = (ln(GNI per capita) - ln(100)) / (ln(75000) - ln(100))
    df['GNI_Cleaned'] = df['GNI per capita'].clip(100, 75000)
    df['Income_Index'] = (np.log(df['GNI_Cleaned']) - np.log(100.0)) / (np.log(75000.0) - np.log(100.0))
    df['Income_Index'] = df['Income_Index'].clip(0, 1)
    
    # D: Calculate geometric mean of the three dimensions for HDI Score
    df['HDI Score'] = (df['Health_Index'] * df['Education_Index'] * df['Income_Index']) ** (1/3)
    df['HDI Score'] = df['HDI Score'].round(3)
    
    # E: Classify country into development tier
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
    
    # Step 6: Label Encoding for Country Name
    print("Performing Label Encoding on Country Name...")
    le = LabelEncoder()
    df['Country_Encoded'] = le.fit_transform(df['Country Name'])
    
    # Save the label encoder for the Flask backend
    with open(encoder_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"Saved LabelEncoder to '{encoder_path}'")
    
    # Step 7: Reorder Columns to align with template.docx specifications
    # Required indexes:
    # 2: Country (encoded)
    # 4: HDI Score (Y)
    # 5: Life expectancy
    # 6: Expected yrs of schooling
    # 7: Mean yrs of schooling
    # 8: GNI per capita
    ordered_columns = [
        'Rank',                     # Index 0
        'Country Name',             # Index 1
        'Country_Encoded',          # Index 2 (Y-independent)
        'Country ISO',              # Index 3
        'HDI Score',                # Index 4 (Y-dependent target)
        'Life expectancy',          # Index 5 (Y-independent)
        'Expected yrs of schooling',# Index 6 (Y-independent)
        'Mean yrs of schooling',    # Index 7 (Y-independent)
        'GNI per capita',           # Index 8 (Y-independent)
        'Development Tier',         # Index 9
        'Health_Index',             # Index 10
        'Expected_Schooling_Index', # Index 11
        'Mean_Schooling_Index',     # Index 12
        'Education_Index',          # Index 13
        'GNI_Cleaned',              # Index 14
        'Income_Index'              # Index 15
    ]
    
    df = df[ordered_columns]
    
    # Save the processed dataset
    df.to_csv(local_processed_path, index=False)
    print(f"\nSaved processed dataset to '{local_processed_path}' (Shape: {df.shape})")
    print(df.iloc[:5, :10])
    return True

if __name__ == "__main__":
    download_and_preprocess()
