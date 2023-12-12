import logging
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.model import EmailClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the saved model and validation dataset
MODEL_PATH = './../model_artifacts/lr/email_classifier.pkl'
VALIDATION_DATASET_PATH = './../data/validation_batch/validation_batch.csv'


def load_validation_data(filepath):
    """Load validation data from a CSV file."""
    data = pd.read_csv(filepath)
    X_val = data['text']
    y_val = data['label']
    return X_val, y_val


def validate_model(model_path, validation_data_path):
    """Load the model and validate it on the provided dataset."""
    logger.info("Starting model validation")
    model = EmailClassifier.load_model(model_path)
    X_val, y_val = load_validation_data(validation_data_path)
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions, average='weighted')
    recall = recall_score(y_val, predictions, average='weighted')
    f1 = f1_score(y_val, predictions, average='weighted')
    logger.info(f"Validation completed at {datetime.now()}")
    logger.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")


if __name__ == "__main__":
    validate_model(MODEL_PATH, VALIDATION_DATASET_PATH)
