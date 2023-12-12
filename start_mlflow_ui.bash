MLFLOW_TRACKING_URI="sqlite:///logging/mlruns.db"
mlflow ui --port 5000 --backend-store-uri ${MLFLOW_TRACKING_URI}
