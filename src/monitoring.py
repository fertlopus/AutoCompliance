# monitoring.py
import mlflow
from model import EmailClassifier
import logging
import sys


def log_experiment(run_name, params, X_train, y_train, X_test, y_test):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        mlflow.set_experiment(run_name)
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)

            # Train and evaluate
            classifier = EmailClassifier(**params)
            classifier.train(X_train, y_train)
            metrics = classifier.evaluate(X_test, y_test)

            # Log metrics and model
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(classifier.model, "model")

            logger.info(f"Experiment logged under run name: {run_name}")

    except Exception as e:
        logger.error(f"Experiment logging failed: {e}")
        sys.exit(1)
