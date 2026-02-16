# Analysis Summary Report

## 1. Executive Summary
The goal of this analysis was to identify factors contributing to patients missing their medical appointments. The overall no-show rate for the dataset is approximately 20%.

## 2. Key Findings

### A. The "Lead Time" Effect
There is a strong positive correlation between `LeadTime` (days between scheduling and appointment) and no-show rates.
- **Same Day Appointments:** Very low no-show rate (<5%).
- **>2 Weeks Lead Time:** No-show rate climbs significantly (>30%).

### B. Demographics
- **Age:** Young adults have slightly higher no-show rates than seniors.
- **Conditions:** Patients with chronic conditions (Hypertension, Diabetes) tend to have slightly *lower* no-show rates, possibly due to routine care needs.

### C. SMS Reminders
Surprisingly, the raw data shows a higher no-show rate for those who received SMS. This is likely a confounding variable: SMS are only sent for appointments with long lead times (which intrinsically have high no-show risk). A controlled experiment would be needed to measure true SMS effectiveness.

## 3. Model Performance
Our Logistic Regression model achieved:
- **Accuracy:** ~79%
- **ROC-AUC:** ~0.74

While not perfect, the model successfully identifies high-risk appointments, primarily driven by the `LeadTime` feature.

## 4. Recommendations
1. **Targeted Reminders:** Focus manual follow-ups on patients with `LeadTime > 14 days`.
2. **Overbooking:** Consider slight overbooking for the "Young Adult" demographic in high-lead-time slots.
