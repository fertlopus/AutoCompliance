import logging
import sys
from data_loader import load_data_in_chunks, preprocess_data, split_data
from utils import feature_extraction
from model import EmailClassifier
import pandas as pd
from datetime import datetime
from joblib import dump
import mlflow


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        # TODO: I will hardcode the path to the dataset
        data_chunks = load_data_in_chunks('./../data/raw/complaints_processed.csv')
        processed_data = pd.concat([preprocess_data(chunk) for chunk in data_chunks])

        # Generate a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"email_classifier_{timestamp}.pkl"
        vectorizer_filename = f"tfidf_vectorizer_{timestamp}.pkl"

        # Feature extraction
        logger.info("Extracting features...")

        tfidf_vectorizer, features_df = feature_extraction(processed_data['narrative'])
        dump(tfidf_vectorizer, f"../model_artifacts/lr/{vectorizer_filename}")

        # Separate features and labels
        X = features_df  # Features from TF-IDF
        y = processed_data['product']  # Labels

        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)

        # MLflow tracking
        mlflow.set_tracking_uri("file:///mlruns") # mlflow ui --backend-store-uri file:///mlruns
        mlflow.set_experiment("Email_Classification_LR")
        with mlflow.start_run():
            logger.info("Training model...")
            classifier = EmailClassifier(max_iter=1000)
            classifier.train(X_train, y_train, save_path = f"./../model_artifacts/lr/{model_filename}")
            # Log parameters, metrics, and model
            mlflow.log_params(classifier.get_params())
            metrics = classifier.evaluate(X_test, y_test)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(classifier.model, "model")
            logger.info(f"Model evaluation metrics: {metrics}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
