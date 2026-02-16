# Hospital No-Show Analysis

## 📌 Project Overview
This project analyzes medical appointment data to understand why patients miss their appointments ("no-shows"). By identifying key drivers, we aim to help hospitals improve scheduling and reduce revenue loss.

## 📂 Dataset
- **Source:** Kaggle (Medical Appointment No Shows)
- **Records:** ~110k appointments
- **Key Features:** Patient demographics, health conditions, scheduling location, and time.

## 🛠️ Components

### 1. Data Pipeline
The project follows a standard modular workflow in the `src/` folder:
- **`data_cleaning.py`**: Handles missing values, duplicates, and correct data types.
- **`feature_engineering.py`**: Creates new features like `LeadTime` (days between scheduling and appointment) and `AgeGroup`.
- **`model_training.py`**: Trains a Logistic Regression model to predict no-shows.
- **`evaluation.py`**: Generates performance metrics and plots (ROC Curve, Confusion Matrix).

### 2. Modeling
We use a **Logistic Regression** baseline.
- **Accuracy:** ~75-80%
- **Key Insight:** `LeadTime` is the strongest predictor. Patients booking >2 weeks in advance are much more likely to miss appointments.

### 3. Dashboard
A Power BI dashboard (`powerbi/hospital_no_show_dashboard.pbix`) visualizes the No-Show rates by:
- Age Group
- Neighborhood
- Day of the Week

## 🚀 How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Pipeline**
   ```bash
   # Clean Data
   python src/data_cleaning.py
   
   # Create Features
   python src/feature_engineering.py
   
   # Train Model
   python src/model_training.py
   
   # Evaluate
   python src/evaluation.py
   ```

3. **Check Reports**
   - Model metrics are printed to console.
   - Plots are saved in `reports/figures/`.

## 📊 Results Summary
- **No-Show Rate:** ~20%
- **Top Factor:** Time between booking and appointment.
- **Recommendation:** Send SMS reminders for appointments scheduled >10 days in advance.

---
*Created by [Your Name] - Data Analyst Portfolio*
