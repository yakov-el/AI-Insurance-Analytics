
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. הגדרות וקבועים
# ----------------------------------------------------------------------
def Synth_data_generation():
    N_POLICIES = 2000 # ~2k policies
    N_MONTHS = 12     # 12 months of observation
    START_DATE = datetime(2023, 1, 1)
    DRIFT_DATE = datetime(2023, 7, 1) # Simple drift after this date
    np.random.seed(42)

    policy_ids = np.arange(1000, 1000 + N_POLICIES)
    months = pd.date_range(start=START_DATE, periods=N_MONTHS, freq='MS')

    # ----------------------------------------------------------------------
    # 1. יצירת נתונים ראשוניים קבועים יחסית (Cross-Sectional Base)
    # ----------------------------------------------------------------------

    initial_data = pd.DataFrame({
        "policy_id": policy_ids,
        "age_start": np.random.normal(40, 10, N_POLICIES).astype(int),
        "region": np.random.choice(["North", "Central", "South", "Coast"], size=N_POLICIES, p=[0.2, 0.4, 0.2, 0.2]),
        "has_agent": np.random.choice([0, 1], size=N_POLICIES, p=[0.3, 0.7]),
        "is_smoker": np.random.choice([0, 1], size=N_POLICIES, p=[0.75, 0.25]),
        "dependents": np.random.randint(0, 5, N_POLICIES),
        "coverage": np.random.choice([10000, 20000, 50000, 100000], size=N_POLICIES, p=[0.4, 0.3, 0.2, 0.1])
    })
    initial_data['age_start'] = np.clip(initial_data['age_start'], 18, 70)

    # ----------------------------------------------------------------------
    # 2. הרחבה ל-Panel Data חודשי והוספת Drift
    # ----------------------------------------------------------------------

    all_data = []
    for i, current_month_date in enumerate(months):
        temp_df = initial_data.copy()
        temp_df['date'] = current_month_date
        temp_df['month'] = temp_df['date'].dt.month

        # Tenure
        temp_df['tenure_m'] = np.random.randint(1, 60, N_POLICIES) + i

        # Age
        temp_df['age'] = temp_df['age_start'] + (temp_df['tenure_m'] // 12)

        # Premium
        base_premium = (temp_df['age'] * 4) + (temp_df['coverage'] / 2000)

        temp_df['premium'] = (base_premium * (1 + 0.5 * temp_df['is_smoker'] - 0.1 * temp_df['has_agent']) + 
                              np.random.normal(0, 10, N_POLICIES)).round(2)
        temp_df['premium'] = np.clip(temp_df['premium'], 50, 500)

        # Drift
        if current_month_date >= DRIFT_DATE:
            temp_df['premium'] = (temp_df['premium'] * 1.15).round(2)

        all_data.append(temp_df)

    df_panel = pd.concat(all_data).sort_values(['policy_id', 'date']).reset_index(drop=True)
    df_panel.drop(columns=['age_start'], inplace=True)

    # ----------------------------------------------------------------------
    # 3. יצירת עמודת המטרה (lapse_next_3m) - פתרון ללא דליפה
    # ----------------------------------------------------------------------

    # א. הגדרת ההסתברות לנטישה (בסיסית)
    # נטישה גבוהה יותר עבור פרמיה גבוהה, ותק נמוך, וללא סוכן
    base_lapse_prob = (
        0.005 +                                          
        (df_panel['premium'] / df_panel['premium'].max()) * 0.05 + 
        (1 - df_panel['has_agent']) * 0.03 +             
        (df_panel['tenure_m'] < 12) * 0.04 +             
        np.random.normal(0, 0.01, len(df_panel))
    )
    base_lapse_prob = np.clip(base_lapse_prob, 0, 0.15) 

    # 🛑 🛑 תיקון קריטי: יצירת הפיצ'ר lapsed_this_month
    df_panel['lapsed_this_month'] = np.random.rand(len(df_panel)) < base_lapse_prob
    df_panel['lapsed_this_month'] = df_panel['lapsed_this_month'].astype(int)

    # ב. יצירת עמודות עזר לבדיקת נטישה קדימה (t+1, t+2, t+3)
    df_panel['lapsed_t_plus_1'] = df_panel.groupby('policy_id')['lapsed_this_month'].shift(-1).fillna(0)
    df_panel['lapsed_t_plus_2'] = df_panel.groupby('policy_id')['lapsed_this_month'].shift(-2).fillna(0)
    df_panel['lapsed_t_plus_3'] = df_panel.groupby('policy_id')['lapsed_this_month'].shift(-3).fillna(0)

    # ג. יצירת TARGET: אם נטישה מתרחשת באחד משלושת החודשים הבאים
    df_panel['lapse_next_3m'] = (
        (df_panel['lapsed_t_plus_1'] == 1) | 
        (df_panel['lapsed_t_plus_2'] == 1) | 
        (df_panel['lapsed_t_plus_3'] == 1)
    ).astype(int)

    # ד. ניקוי עמודות העזר
    df_panel.drop(columns=['lapsed_this_month', 'lapsed_t_plus_1', 
                          'lapsed_t_plus_2', 'lapsed_t_plus_3'], inplace=True)

    # ----------------------------------------------------------------------
    # 4. הוספת Leakage Trap (מלכודת דליפה) - חובה לאימות
    # ----------------------------------------------------------------------

    # Leakage Trap: נניח שזה פיצ'ר שקיים רק *אחרי* שהחברה כבר החליטה על שימור
    df_panel['post_event_retention_effort'] = np.where(
        # אם תתרחש נטישה ב-3 החודשים הבאים, נניח שהיה מאמץ שימור גבוה שנסגר
        df_panel.groupby('policy_id')['lapse_next_3m'].shift(-3).fillna(0).astype(bool), 
        np.random.normal(5, 2, len(df_panel)), 
        0
    )
    df_panel['post_event_retention_effort'] = np.clip(df_panel['post_event_retention_effort'], 0, 10)


    # ----------------------------------------------------------------------
    # 5. ניקוי וסיום (הכנה לאימון)
    # ----------------------------------------------------------------------

    df_final = df_panel.sort_values(['policy_id', 'date']).reset_index(drop=True)

    print("Data Generation Complete")
    lapse_rate = df_final['lapse_next_3m'].mean()
    active_rate = 1 - lapse_rate

    print(f"✅ Class Balance Check:")
    print(f"Total Rows: {len(df_final)}")
    print(f"Lapsed Policies (Target=1): {lapse_rate:.2%} ({int(lapse_rate * len(df_final))} rows)")
    print(f"Active Policies (Target=0): {active_rate:.2%} ({int(active_rate * len(df_final))} rows)")
    return df_final

# ניתן להריץ את הפונקציה כדי לוודא שאין שגיאות
df_synthetic = Synth_data_generation()



