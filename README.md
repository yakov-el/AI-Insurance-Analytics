AI Insurance Analytics Pipeline

This project implements a complete end-to-end pipeline for insurance analytics, combining predictive modeling with retrieval-augmented generation (RAG).
It includes:

Synthetic data generation for insurance policy behavior

Training a churn-prediction model (LGBM/XGBoost) using proper temporal splits

RAG-based recommendations for lapse-prevention and lead-conversion strategies

⚙️ Prerequisites

The project uses standard Python libraries.
It is recommended to run it inside a virtual environment.

pip install -r requirements.txt

🚀 How to Run

Execute the full pipeline with a single command:

python run.py

📂 Outputs

Running the pipeline creates an out/ directory (if it doesn’t exist) and populates it with:

out/model.pkl — Trained model (with preprocessing) wrapped in a Pipeline

out/metrics.json — Model performance metrics on the test set

out/shap_plot.png — Global SHAP feature-importance visualization

out/RAG_lapse_prevention_plans.json — Generated action plans for lapse prevention

out/RAG_lead_conversion_plans.json — Generated action plans for lead conversion

🏛️ Project Structure

run.py — Main orchestrator script for the entire workflow

data_generator.py — Synthetic data generation and temporal splitting

model_train.py — Training, tuning, and evaluating predictive models

rag_module.py — RAG logic: retrieval and generation of action plans

DISCUSSION.md — Design considerations, architecture notes, and reasoning

corpus_lapse/ — Source documents for lapse-prevention recommendations

corpus_leads/ — Source documents for lead-conversion recommendations

requirements.txt — Dependency list
