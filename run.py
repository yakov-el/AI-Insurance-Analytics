#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
from data_generator import Synth_data_generation
from model_train import perform_temporal_split_and_prep, train_and_evaluate_model
from rag_module import rag_analysis
import json
import os
import pandas as pd # ודא ש-pandas מיובא אם אתה משתמש בפונקציות שלו כאן

# -----------------------------------------------------------
# פונקציית הרצה ראשית
# -----------------------------------------------------------
def main():
    """
    מריץ את כל ה-Pipeline של המטלה:
    1. יצירת נתונים
    2. פיצול ועיבוד
    3. אימון מודל
    4. הפעלת RAG
    """
    start_time = time.time()

    # -----------------------------------------------------------
    # הגדרת סביבה (בדיקה ש-API KEY קיים)
    # -----------------------------------------------------------
    if not os.environ.get("GEMINI_API_KEY"):
        print("=============================================================")
        print("⚠️ אזהרה: משתנה הסביבה GEMINI_API_KEY אינו מוגדר.")
        print("שלב ה-RAG (שלב 4) ירוץ, אך יחזיר תגובות מדוּמוֹת (mocked LLM responses).")
        print("=============================================================")

    # 1️⃣ יצירת דאטה סינתטי
    print("1️⃣ מתחיל יצירת נתונים...")
    df = Synth_data_generation()
    print(f"   נוצרו {len(df)} רשומות.")

    # 2️⃣ פיצול ועיבוד נתונים
    print("\n2️⃣ מפצל ומעבד נתונים...")
    (X_train, y_train, X_val, y_val, 
     X_test, y_test, df_test_raw, preprocessor) = perform_temporal_split_and_prep(df)
    print(f"   גודל סט אימון: {len(X_train)}")

    # 3️⃣ אימון מודל ובדיקה (ML Ops)
    print("\n3️⃣ מאמן ומעריך מודל...")
    model, df_rag, metrics = train_and_evaluate_model(
        X_train, y_train, X_val, y_val, X_test, y_test, df_test_raw, preprocessor
    )
    print(f"   המודל אומן בהצלחה. סוג מודל: {type(model.named_steps['classifier']).__name__}")

    # הדפסת מדדי הביצוע
    metrics_path = os.path.join('out', 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            saved_metrics = json.load(f)
            print("\n✅ מדדי המודל הסופיים (מתוך metrics.json):")
            for key, value in saved_metrics.items():
                print(f"   - {key}: {value:.4f}")

    # 4️⃣ הפעלת RAG (שליפה ויצירה מוגברת)
    print("\n4️⃣ מתחיל ניתוח RAG (Retrieval Augmented Generation)...")
    rag_output = rag_analysis(df_rag, metrics)

    print("\n--- Pipeline RAG הושלם ---")
    print("פלט ה-RAG נשמר ב-out/rag_output.txt")

    # הדפסת סיכום מהיר של התוכנית הראשונה
    first_plan_start = rag_output.find("--- 3-Step Retention Plan ---")
    first_plan_end = rag_output.find("*** Customer Profile: Median Risk")

    if first_plan_start != -1:
        summary_text = rag_output[first_plan_start:first_plan_end if first_plan_end != -1 else len(rag_output)].strip()
        print("\nסיכום התוכנית הראשונה שנוצרה (לקוח בסיכון גבוה):")
        print(summary_text)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n🏁 התהליך הסתיים בהצלחה. סך זמן ריצה: {total_time:.2f} שניות.")

# -----------------------------------------------------------
# הרצת הפונקציה הראשית
# -----------------------------------------------------------
if __name__ == "__main__":
    main()


# In[ ]:




