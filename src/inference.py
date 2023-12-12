import pandas as pd
import logging
from .model import EmailClassifier
import joblib


class EmailPredictor:
    """
    A class for handling the loading and inference of the email classification model.
    """

    def __init__(self, model_path: str, vectorizer_path: str) -> None:
        """
        Initialize the EmailPredictor with the path to the trained model.

        Args:
        model_path (str): Path to the trained model file.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = self._load_model(model_path)
        self.vectorizer = self._load_vectorizer(vectorizer_path)

    def _load_model(self, model_path: str) -> EmailClassifier:
        """
        Load the trained email classification model.

        Args:
        model_path (str): Path to the trained model file.

        Returns:
        EmailClassifier: The loaded model.
        """
        try:
            return joblib.load(model_path)
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def _load_vectorizer(self, vectorizer_path: str):
        try:
            return joblib.load(vectorizer_path)
        except Exception as e:
            self.logger.error(f"Failed to load vectorizer from {vectorizer_path}: {e}")
            raise

    def preprocess_input(self, text: str) -> pd.DataFrame:
        """
        Preprocess the input text for prediction.

        Args:
        text (str): The input text to preprocess.

        Returns:
        pd.DataFrame: A DataFrame containing the preprocessed text.
        """
        try:
            # TODO: Implement preprocessing steps similar to those used during training
            processed_text = self.vectorizer.transform([text])
            return pd.DataFrame(processed_text.toarray())
        except Exception as e:
            self.logger.error(f"Preprocessing failed for text: {text}: {e}")
            raise

    def predict(self, text: str) -> str:
        """
        Predict the class of a given email text.

        Args:
        text (str): The email text to classify.

        Returns:
        str: The predicted label.
        """
        try:
            preprocessed_text = self.preprocess_input(text)
            prediction = self.model.predict(preprocessed_text)[0]
            return prediction
        except Exception as e:
            self.logger.error(f"Prediction failed for text: {text}: {e}")
            raise
