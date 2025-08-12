cd Train/Walmart_new
uvicorn comparison_gru_vs_ensemble_api:app --host=0.0.0.0 --port=$PORT
