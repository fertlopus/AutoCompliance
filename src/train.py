import logging
import sys
from data_loader import load_data_in_chunks, preprocess_data, split_data
from utils import feature_extraction
from model import EmailClassifier
import pandas as pd
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

        # Feature extraction
        logger.info("Extracting features...")
        X, y = feature_extraction(processed_data['narrative']), processed_data['product']

        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)

        # MLflow tracking
        mlflow.set_experiment("Email_Classification_LR")
        with mlflow.start_run():
            logger.info("Training model...")
            classifier = EmailClassifier()
            classifier.train(X_train, y_train)

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
