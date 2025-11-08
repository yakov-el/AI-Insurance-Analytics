# import pandas as pd
# import numpy as np
# import os
# import json
# import joblib
# import shap
# import lightgbm as lgb
# import matplotlib.pyplot as plt
# from datetime import datetime
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
# from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
# from sklearn.pipeline import Pipeline
# # ----------------------------------------------------------------------
# # 0. הגדרות וקבועים
# # ----------------------------------------------------------------------

# OUT_DIR = 'out'
# os.makedirs(OUT_DIR, exist_ok=True)

# # תאריכים לפיצול זמני
# TRAIN_END_DATE = datetime(2023, 9, 1) # עד ספטמבר (לא כולל)
# VALIDATION_END_DATE = datetime(2023, 10, 1) # ספטמבר עבור Validation, אוקטובר והלאה ל-Test
# SEED = 42 # דטרמיניזם

# # ----------------------------------------------------------------------
# # 1. פונקציות עזר לחישוב מטריקות
# # ----------------------------------------------------------------------

# def calculate_precision_at_k(y_true, y_probas, k_percent):
#     """
#     מחשב Precision@k (היעילות ב-k% המדורגים ראשונים).
#     """
#     k = int(len(y_probas) * k_percent / 100)
#     if k == 0:
#         return 0.0

#     df = pd.DataFrame({'proba': y_probas, 'target': y_true})
#     df_sorted = df.sort_values(by='proba', ascending=False)
    
#     top_k = df_sorted.head(k)
    
#     precision = top_k['target'].mean()
    
#     return precision

# # ----------------------------------------------------------------------
# # 2. פיצול זמני ועיבוד מקדים
# # ----------------------------------------------------------------------

# def perform_temporal_split_and_prep(df_final: pd.DataFrame):
#     """
#     מבצע פיצול זמני מחמיר ל-Train, Validation ו-Test
#     ומכין את הנתונים לאימון המודל.
#     """
    
#     # 2.1 פיצול זמני (דרישה 2: Temporal Integrity)
#     df_train = df_final[df_final['date'] < TRAIN_END_DATE].copy()
#     df_val_test = df_final[df_final['date'] >= TRAIN_END_DATE].copy()
#     df_val = df_val_test[df_val_test['date'] < VALIDATION_END_DATE].copy()
#     df_test_raw = df_val_test[df_val_test['date'] >= VALIDATION_END_DATE].copy()
    
#     # ----------------------------------------
#     # 2.2 הגדרת עמודות - Data Leakage Safeguard (דרישה 1)
#     # ----------------------------------------
#     DROP_COLS = ['policy_id', 'date', 'post_event_retention_effort']
#     CATEGORICAL_COLS = ['region'] # 'has_agent' ו 'is_smoker' כבר 0/1
#     TARGET_COL = 'lapse_next_3m'
    
#     # עמודות הפיצ'רים שיישארו
#     FEATURE_COLS = [col for col in df_train.columns if col not in DROP_COLS + [TARGET_COL]]

#     # 2.3 עיבוד מקדים (One-Hot Encoding ל-'region')
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
#         ],
#         remainder='passthrough'
#     )
    
#     # 2.4 הפרדת X ו-Y והפעלת ה-Pipeline
#     def get_X_y(df, preprocessor_obj, fit=False):
#         X_raw = df[FEATURE_COLS]
#         y = df[TARGET_COL]
        
#         if fit:
#             preprocessor_obj.fit(X_raw)
            
#         X_prep = preprocessor_obj.transform(X_raw)
        
#         # המרת מטריצת NumPy בחזרה ל-DataFrame עם שמות עמודות נכונים
#         # שמות הפיצ'רים שאינם קטגוריאליים
#         passthrough_cols = [col for col in X_raw.columns if col not in CATEGORICAL_COLS]
#         # שמות הפיצ'רים החדשים שנוצרו (מה-OneHot)
#         cat_feature_names = preprocessor_obj.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_COLS)
        
#         feature_names = list(cat_feature_names) + passthrough_cols
        
#         X_df = pd.DataFrame(X_prep, columns=feature_names, index=df.index)
        
#         return X_df, y

#     # התאמת המעבד המקדים ל-Train
#     X_train_df, y_train = get_X_y(df_train, preprocessor, fit=True)
#     X_val_df, y_val = get_X_y(df_val, preprocessor)
#     X_test_df, y_test = get_X_y(df_test_raw, preprocessor)
    
#     print("\n✅ Temporal Split and Preprocessing Complete.")
#     print(f"Train set: {len(X_train_df)} rows (End Date: {df_train['date'].max()})")
#     print(f"Validation set: {len(X_val_df)} rows (Period: {df_val['date'].min()} to {df_val['date'].max()})")
#     print(f"Test set: {len(X_test_df)} rows (Start Date: {df_test_raw['date'].min()})")
    
#     # מחזיר גם את df_test_raw לשימוש ב-RAG (כדי לקחת דוגמאות לקוחות)
#     # 🚨 תיקון: הוספת preprocessor לרשימת המשתנים המוחזרים (כדי שיהיו 8)
#     return X_train_df, y_train, X_val_df, y_val, X_test_df, y_test, df_test_raw, preprocessor

# # ----------------------------------------------------------------------
# # 3. אימון, כוונון והערכה (Model_train)
# # ----------------------------------------------------------------------

# def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, df_test_raw, preprocessor):
#     """
#     כוונון קל באמצעות RandomizedSearch על TRAIN בלבד (עם TimeSeriesSplit),
#     אימון סופי עם early stopping על VAL, הערכה על TEST,
#     שמירת מודל, שמירת metrics.json ויצירת SHAP bar plot.
#     """

#     # ---- 1) הגדרת מודל בסיסי ופרמטרים ----
#     lgbm = lgb.LGBMClassifier(random_state=SEED, class_weight='balanced', verbose=-1)

#     param_grid = {
#         'n_estimators': [100, 200, 300],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'num_leaves': [15, 31, 63],
#         'max_depth': [3, 5, 7],
#     }

#     # ---- 2) RandomizedSearch: השתמש ב-TRAIN בלבד עם TimeSeriesSplit (דרישה 2) ----
#     print("\n🔬 Performing Light Tuning (Randomized Search) on TRAIN only...")
#     tscv = TimeSeriesSplit(n_splits=2)
#     search = RandomizedSearchCV(
#         lgbm, param_grid, n_iter=30, scoring='average_precision', # AUC-PR כמדד
#         cv=tscv, random_state=SEED, n_jobs=-1, verbose=0
#     )
#     search.fit(X_train, y_train)
#     best_params = search.best_params_
#     print(f"🏆 Best Hyperparameters found (train-only CV): {best_params}")

#     # ---- 3) אימון סופי עם Early Stopping על סט ה-Validation ----
#     final_model = lgb.LGBMClassifier(
#         **best_params, # 🚨 תיקון שגיאת תחביר: הסרת פסיק מיותר
#         random_state=SEED,
#         class_weight='balanced',
#         verbose=-1
#     )

#     final_model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         eval_metric='average_precision',
#         callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
#     )

#     # ---- 4) הערכה על Test ----
#     y_test_probas = final_model.predict_proba(X_test)[:, 1]
#     auc_pr = average_precision_score(y_test, y_test_probas)
#     roc_auc = roc_auc_score(y_test, y_test_probas)
#     p_at_1 = calculate_precision_at_k(y_test, y_test_probas, 1)
#     p_at_5 = calculate_precision_at_k(y_test, y_test_probas, 5)

#     metrics = {
#         "AUC-PR": round(auc_pr, 4),
#         "ROC_AUC": round(roc_auc, 4),
#         "Precision@1%": round(p_at_1, 4),
#         "Precision@5%": round(p_at_5, 4)
#     }

#     print("\n📊 Model Evaluation (Test Set):")
#     print(f"Primary Metric (AUC-PR): {metrics['AUC-PR']:.4f}")
#     print(f"ROC AUC: {metrics['ROC_AUC']:.4f}")
#     print(f"Precision@1%: {metrics['Precision@1%']:.4f}")
#     print(f"Precision@5%: {metrics['Precision@5%']:.4f}")

#     # ---- 5) שמירת מודל ומטריקות ----
#     model_path = os.path.join(OUT_DIR, 'model.pkl')
#     # שמירת ה-pipeline המלא (כולל preprocessor)
#     full_pipeline = Pipeline([
#         ('preprocessor', preprocessor),
#         ('classifier', final_model)
#     ])
#     joblib.dump(full_pipeline, model_path)
#     print(f"\n💾 Model pipeline saved to {model_path}")

#     metrics_path = os.path.join(OUT_DIR, 'metrics.json')
#     with open(metrics_path, 'w') as f:
#         json.dump(metrics, f, indent=4)
#     print(f"💾 Metrics saved to {metrics_path}")

#     # ---- 6) SHAP: גלובלי (bar) ושמירה כ-PNG ----
#     try:
#         explainer = shap.TreeExplainer(final_model)
#         shap_values = explainer.shap_values(X_test)
        
#         if isinstance(shap_values, list):
#             shap_for_positive = shap_values[1]
#         else:
#             shap_for_positive = shap_values

#         # יצירת summary bar plot ושמירה
#         shap.summary_plot(shap_for_positive, X_test, plot_type="bar", show=False)
#         shap_path = os.path.join(OUT_DIR, 'shap_plot.png')
#         plt.savefig(shap_path, bbox_inches='tight')
#         plt.close()
#         print(f"💾 SHAP Plot saved to {shap_path}")
#     except Exception as e:
#         print(f"⚠️ SHAP Plot failed: {e}. (Make sure X_test has non-zero size.)")

#     # ---- 7) הכנת טסט ל-RAG ----
#     df_test_for_rag = df_test_raw.copy()
#     df_test_for_rag['lapse_proba'] = y_test_probas

#     return full_pipeline, df_test_for_rag, metrics
import pandas as pd
import numpy as np
import os
import json
import joblib
import shap
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, precision_score
from sklearn.pipeline import Pipeline
from scipy.stats import randint as sp_randint, uniform as sp_uniform

# ----------------------------------------------------------------------
# 0. הגדרות וקבועים
# ----------------------------------------------------------------------

OUT_DIR = 'out'
os.makedirs(OUT_DIR, exist_ok=True)

# תאריכים לפיצול זמני
TRAIN_END_DATE = datetime(2023, 9, 1) # עד ספטמבר (לא כולל)
VALIDATION_END_DATE = datetime(2023, 10, 1) # ספטמבר עבור Validation, אוקטובר והלאה ל-Test
SEED = 42 # דטרמיניזם

# ----------------------------------------------------------------------
# 1. פונקציות עזר לחישוב מטריקות
# ----------------------------------------------------------------------

def calculate_precision_at_k(y_true, y_probas, k_percent):
    """
    מחשב Precision@k (היעילות ב-k% המדורגים ראשונים).
    """
    k = int(len(y_probas) * k_percent / 100)
    if k == 0:
        return 0.0

    df = pd.DataFrame({'proba': y_probas, 'target': y_true})
    df_sorted = df.sort_values(by='proba', ascending=False)
    
    top_k = df_sorted.head(k)
    
    precision = top_k['target'].mean()
    
    return precision

# ----------------------------------------------------------------------
# 2. פיצול זמני ועיבוד מקדים
# ----------------------------------------------------------------------

def perform_temporal_split_and_prep(df_final: pd.DataFrame):
    """
    מבצע פיצול זמני מחמיר ל-Train, Validation ו-Test
    ומכין את הנתונים לאימון המודל.
    """
    
    # 2.1 פיצול זמני (דרישה 2: Temporal Integrity)
    df_train = df_final[df_final['date'] < TRAIN_END_DATE].copy()
    df_val_test = df_final[df_final['date'] >= TRAIN_END_DATE].copy()
    df_val = df_val_test[df_val_test['date'] < VALIDATION_END_DATE].copy()
    df_test_raw = df_val_test[df_val_test['date'] >= VALIDATION_END_DATE].copy()
    
    # ----------------------------------------
    # 2.2 הגדרת עמודות - Data Leakage Safeguard (דרישה 1)
    # ----------------------------------------
    DROP_COLS = ['policy_id', 'date', 'post_event_retention_effort']
    CATEGORICAL_COLS = ['region'] # 'has_agent' ו 'is_smoker' כבר 0/1
    TARGET_COL = 'lapse_next_3m'
    
    # עמודות הפיצ'רים שיישארו
    FEATURE_COLS = [col for col in df_train.columns if col not in DROP_COLS + [TARGET_COL]]

    # 2.3 עיבוד מקדים (One-Hot Encoding ל-'region')
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
        ],
        remainder='passthrough'
    )
    
    # 2.4 הפרדת X ו-Y והפעלת ה-Pipeline
    def get_X_y(df, preprocessor_obj, fit=False):
        X_raw = df[FEATURE_COLS]
        y = df[TARGET_COL]
        
        if fit:
            preprocessor_obj.fit(X_raw)
            
        X_prep = preprocessor_obj.transform(X_raw)
        
        # המרת מטריצת NumPy בחזרה ל-DataFrame עם שמות עמודות נכונים
        # שמות הפיצ'רים שאינם קטגוריאליים
        passthrough_cols = [col for col in X_raw.columns if col not in CATEGORICAL_COLS]
        # שמות הפיצ'רים החדשים שנוצרו (מה-OneHot)
        cat_feature_names = preprocessor_obj.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_COLS)
        
        feature_names = list(cat_feature_names) + passthrough_cols
        
        X_df = pd.DataFrame(X_prep, columns=feature_names, index=df.index)
        
        return X_df, y

    # התאמת המעבד המקדים ל-Train
    X_train_df, y_train = get_X_y(df_train, preprocessor, fit=True)
    X_val_df, y_val = get_X_y(df_val, preprocessor)
    X_test_df, y_test = get_X_y(df_test_raw, preprocessor)
    
    print("\n✅ Temporal Split and Preprocessing Complete.")
    print(f"Train set: {len(X_train_df)} rows (End Date: {df_train['date'].max()})")
    print(f"Validation set: {len(X_val_df)} rows (Period: {df_val['date'].min()} to {df_val['date'].max()})")
    print(f"Test set: {len(X_test_df)} rows (Start Date: {df_test_raw['date'].min()})")
    
    # מחזיר גם את df_test_raw לשימוש ב-RAG (כדי לקחת דוגמאות לקוחות)
    return X_train_df, y_train, X_val_df, y_val, X_test_df, y_test, df_test_raw, preprocessor

# ----------------------------------------------------------------------
# 3. אימון, כוונון והערכה (Model_train)
# ----------------------------------------------------------------------


# פונקציה חיצונית חדשה לביצוע כוונון והערכה עבור מודל ספציפי
def tune_and_evaluate_single_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test, preprocessor):
    
    # חישוב משקל קלאסים מאוזן (כדי לתת משקל למיעוט הנוטש)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    if model_name == 'LGBM':
        # שימוש ב-class_weight='balanced'
        model = lgb.LGBMClassifier(random_state=SEED, class_weight='balanced', verbose=-1) 
        param_grid = {
            'n_estimators': sp_randint(100, 400),
            'learning_rate': sp_uniform(0.01, 0.1),
            'num_leaves': sp_randint(15, 63),
            'max_depth': [3, 5, 7, 9],
            'subsample': sp_uniform(0.6, 0.4),
        }
    
    elif model_name == 'XGBoost':
        # שימוש ב-scale_pos_weight
        model = xgb.XGBClassifier(random_state=SEED, eval_metric='logloss', 
                                  use_label_encoder=False, scale_pos_weight=scale_pos_weight) 
        param_grid = {
            'n_estimators': sp_randint(100, 400),
            'learning_rate': sp_uniform(0.01, 0.1),
            'max_depth': [3, 5, 7, 9],
            'subsample': sp_uniform(0.6, 0.4),
            'colsample_bytree': sp_uniform(0.6, 0.4),
        }
        
    else:
        raise ValueError("Unknown model name")

    # ---- 1) RandomizedSearch ----
    print(f"\n🔬 Performing Tuning ({model_name}) on TRAIN only...")
    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        model, param_distributions=param_grid, n_iter=50, 
        scoring='average_precision', cv=tscv, random_state=SEED, n_jobs=-1, verbose=0
    )
    
    # קריאת ה-fit הראשונה: *ללא* fit_params.
    search.fit(X_train, y_train) 
    
    best_params = search.best_params_
    print(f"🏆 Best Hyperparameters found ({model_name} CV): {best_params}")

    # ---- 2) אימון סופי עם Early Stopping על סט ה-Validation ----
    
    # יצירת מודל חדש עם הפרמטרים הטובים ביותר
    final_model = model.__class__(**best_params, random_state=SEED, verbose=-1) 
    
    # יצירת fit_params *רק* עבור final_model.fit
    fit_params = {} 
    
    if model_name == 'LGBM':
        final_model.set_params(class_weight='balanced')
        # פרמטרים ספציפיים ל-LGBM (כאן זה עובד)
        fit_params = {
            'eval_set': [(X_val, y_val)],
            'eval_metric': 'average_precision',
            'callbacks': [lgb.early_stopping(stopping_rounds=20, verbose=False)]
        }
    
    elif model_name == 'XGBoost':
        # הגדרת פרמטרים שהיו במודל המקורי
        final_model.set_params(scale_pos_weight=scale_pos_weight, 
                             eval_metric='logloss',
                             use_label_encoder=False)
        

        
        fit_params = {} # משאירים ריק בכוונה
        # ------------------------------------------------------------------

    # ⚠️ קריאת ה-fit השנייה: *עם* fit_params.
    # עבור XGBoost, fit_params יהיה ריק וזה ימנע את השגיאה.
    final_model.fit(X_train, y_train, **fit_params)

    # ---- 3) הערכה על Test ----
    y_test_probas = final_model.predict_proba(X_test)[:, 1]
    auc_pr = average_precision_score(y_test, y_test_probas)
    roc_auc = roc_auc_score(y_test, y_test_probas)
    p_at_1 = calculate_precision_at_k(y_test, y_test_probas, 1)
    p_at_5 = calculate_precision_at_k(y_test, y_test_probas, 5)

    metrics = {
        "AUC-PR": round(auc_pr, 4),
        "ROC_AUC": round(roc_auc, 4),
        "Precision@1%": round(p_at_1, 4),
        "Precision@5%": round(p_at_5, 4)
    }
    
    return final_model, metrics, y_test_probas


def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, df_test_raw, preprocessor):
    """
    מבצע השוואה בין LGBM ל-XGBoost, בוחר את המודל הטוב ביותר (לפי AUC-PR),
    ושומר את תוצאותיו.
    """
    
    # ---- 1) השוואה בין מודלים ----
    
    # אימון והערכת LGBM
    lgbm_model, lgbm_metrics, lgbm_probas = tune_and_evaluate_single_model(
        'LGBM', X_train, y_train, X_val, y_val, X_test, y_test, preprocessor)
    print(f"\n📊 LGBM Evaluation (Test Set - AUC-PR): {lgbm_metrics['AUC-PR']:.4f}")
    
    # אימון והערכת XGBoost
    xgb_model, xgb_metrics, xgb_probas = tune_and_evaluate_single_model(
        'XGBoost', X_train, y_train, X_val, y_val, X_test, y_test, preprocessor)
    print(f"📊 XGBoost Evaluation (Test Set - AUC-PR): {xgb_metrics['AUC-PR']:.4f}")
    
    # ---- 2) בחירת המודל המנצח ----
    if lgbm_metrics['AUC-PR'] >= xgb_metrics['AUC-PR']:
        final_model = lgbm_model
        metrics = lgbm_metrics
        y_test_probas = lgbm_probas
        model_type = "LGBMClassifier"
        print(f"\n🏆 LGBM ({metrics['AUC-PR']:.4f}) is the winner.")
    else:
        final_model = xgb_model
        metrics = xgb_metrics
        y_test_probas = xgb_probas
        model_type = "XGBClassifier"
        print(f"\n🏆 XGBoost ({metrics['AUC-PR']:.4f}) is the winner.")

    print(f"\n✅ Final Model Type: {model_type}")
    
    # הצגת המדדים של המודל הנבחר
    print("\n📊 Final Model Evaluation (Test Set):")
    for key, value in metrics.items():
        print(f"Primary Metric ({key}): {value:.4f}")

    # ---- 3) שמירת מודל ומטריקות ----
    
    # יצירת Pipeline לשמירה
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', final_model)
    ])
    model_path = os.path.join(OUT_DIR, 'model.pkl')
    joblib.dump(full_pipeline, model_path)
    print(f"\n💾 Model pipeline saved to {model_path}")

    metrics_path = os.path.join(OUT_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"💾 Metrics saved to {metrics_path}")

    # ---- 4) SHAP: גלובלי (bar) ושמירה כ-PNG ----
    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_test)
        
        # טיפול בפורמטים שונים של LGBM / XGBoost
        if isinstance(shap_values, list):
            shap_for_positive = shap_values[1]
        else:
            shap_for_positive = shap_values

        # יצירת summary bar plot ושמירה
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_for_positive, X_test, plot_type="bar", show=False)
        shap_path = os.path.join(OUT_DIR, 'shap_plot.png')
        plt.savefig(shap_path, bbox_inches='tight')
        plt.close()
        print(f"💾 SHAP Plot saved to {shap_path}")
    except Exception as e:
        print(f"⚠️ SHAP Plot failed: {e}. (Make sure X_test has non-zero size.)")

    # ---- 5) הכנת טסט ל-RAG ----
    df_test_for_rag = df_test_raw.copy()
    df_test_for_rag['lapse_proba'] = y_test_probas

    return full_pipeline, df_test_for_rag, metrics