MLFLOW_TRACKING_URI="sqlite:///data/mlruns.db"
mlflow ui --port 5000 --backend-store-uri ${MLFLOW_TRACKING_URI}
