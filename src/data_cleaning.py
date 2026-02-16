import pandas as pd
import os

def load_data(filepath):
    """Loads CSV data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Cleans the raw dataframe:
    - Renames columns to cleaner format
    - Converts dates
    - Drops duplicates/invalid rows
    """
    # 1. Rename Columns
    df.rename(columns={
        'Hipertension': 'Hypertension',
        'Handcap': 'Handicap',
        'No-show': 'NoShow'
    }, inplace=True)
    
    # Standardize column names
    df.columns = [col.strip() for col in df.columns]

    # 2. Convert to Datetime
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

    # 3. Handle Invalid Data
    # Drops Age < 0 and extremely high age (likely errors)
    df = df[(df['Age'] >= 0) & (df['Age'] <= 120)]

    # 4. Drop Duplicates
    # Check for duplicate AppointmentIDs
    df.drop_duplicates(subset=['AppointmentID'], inplace=True)
    
    return df

if __name__ == "__main__":
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "KaggleV2-May-2016.csv")
    PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "no_show_clean.csv")

    try:
        print("Loading raw data...")
        df = load_data(RAW_PATH)
        
        print(f"Original shape: {df.shape}")
        
        print("Cleaning data...")
        df_clean = clean_data(df)
        
        print(f"Cleaned shape: {df_clean.shape}")
        
        # Save
        os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
        df_clean.to_csv(PROCESSED_PATH, index=False)
        print(f"Data saved to {PROCESSED_PATH}")
        
    except Exception as e:
        print(f"Error: {e}")
