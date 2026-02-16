import pandas as pd
import numpy as np
import os

def create_features(df):
    """
    Creates new features for analysis:
    - LeadTime (Days between scheduling and appointment)
    - AgeGroup
    - IsWeekend
    - ChronicCondition count
    """
    df = df.copy()
    
    # 1. Lead Time (Wait Days)
    # Normalize to midnight to avoid negative days due to time diffs
    df['LeadTime'] = (df['AppointmentDay'].dt.normalize() - df['ScheduledDay'].dt.normalize()).dt.days
    
    # Clamp negative lead time to 0 (same day booking)
    df['LeadTime'] = df['LeadTime'].apply(lambda x: max(x, 0))

    # 2. Age Groups
    bins = [0, 12, 18, 55, 120]
    labels = ['Child', 'Teen', 'Adult', 'Senior']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    # 3. Appointment Day of Week
    df['DayOfWeek'] = df['AppointmentDay'].dt.day_name()
    
    # 4. Chronic Condition Flag (Sum of binary conditions)
    conditions = ['Hypertension', 'Diabetes', 'Alcoholism', 'Handicap']
    df['ChronicConditions'] = df[conditions].sum(axis=1)
    df['HasCondition'] = (df['ChronicConditions'] > 0).astype(int)

    # 5. Encode Target
    df['NoShowBinary'] = df['NoShow'].map({'Yes': 1, 'No': 0})
    
    return df

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "no_show_clean.csv")
    OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "no_show_features.csv")

    try:
        print("Loading clean data...")
        if not os.path.exists(INPUT_PATH):
            raise FileNotFoundError("Please run data_cleaning.py first.")
            
        df = pd.read_csv(INPUT_PATH, parse_dates=['ScheduledDay', 'AppointmentDay'])
        
        print("Engineering features...")
        df_features = create_features(df)
        
        # Save
        df_features.to_csv(OUTPUT_PATH, index=False)
        print(f"Feature-rich data saved to {OUTPUT_PATH}")
        print(df_features[['LeadTime', 'AgeGroup', 'NoShowBinary']].head())

    except Exception as e:
        print(f"Error: {e}")
