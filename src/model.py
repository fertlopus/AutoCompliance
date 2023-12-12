import logging
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from typing import Any, Dict, Optional, Tuple


class EmailClassifier:
    """
    A classifier for email classification using Logistic Regression.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = LogisticRegression(**kwargs)

    def train(self,
              X_train: Any,
              y_train: Any,
              param_grid: Optional[Dict[str, Any]] = None,
              save_path: Optional[str] = None) -> None:
        """
        Training the Logistic Regression model with optional hyperparameter tuning.

        Args:
        X_train: Feature data for training.
        y_train: Target labels for training.
        param_grid (Optional[Dict[str, Any]]): Parameters for hyperparameter tuning.
        save_path (Optional[str]): Path to save the trained model.
        """
        try:
            if param_grid:
                self.model = GridSearchCV(self.model, param_grid, cv=5)
            self.model.fit(X_train, y_train)

            if isinstance(self.model, GridSearchCV):
                self.logger.info(f"Best parameters: {self.model.best_params_}")
                self.model = self.model.best_estimator_

            self.logger.info("Model training completed.")
            self._log_model_metrics(X_train, y_train, "Training")

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                joblib.dump(self.model, save_path)
                self.logger.info(f"Model saved at {save_path}")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def predict(self, X: Any) -> Any:
        """
        Make predictions using the trained Logistic Regression model.

        Args:
        X: Feature data for making predictions.

        Returns:
        Predicted labels.
        """
        try:
            return self.model.predict(X)
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def evaluate(self, X_test: Any, y_test: Any) -> Dict[str, Any]:
        """
        Evaluate the model's performance on the test set.

        Args:
        X_test: Feature data for testing.
        y_test: True labels for testing.

        Returns:
        A dictionary containing accuracy and classification report.
        """
        try:
            predictions = self.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)
            self._log_model_metrics(X_test, y_test, "Testing")
            return {"accuracy": accuracy, "report": report}
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise

    @staticmethod
    def load_model(load_path: str) -> "EmailClassifier":
        """
        Load a saved Logistic Regression model.

        Args:
        load_path (str): Path to the saved model.

        Returns:
        An instance of EmailClassifier with the loaded model.
        """
        try:
            model = joblib.load(load_path)
            return EmailClassifier(model=model)
        except Exception as e:
            logging.error(f"Loading model failed: {e}")
            raise

    def _log_model_metrics(self, X: Any, y: Any, phase: str) -> None:
        """
        Log model metrics for a given dataset phase.

        Args:
        X: Feature data.
        y: True labels.
        phase (str): The phase of the model ('Training' or 'Testing').
        """
        # TODO: Implement logging of metrics
        # This function can be extended to integrate with MLflow or other tracking systems
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        self.logger.info(f"{phase} Accuracy: {accuracy}")
