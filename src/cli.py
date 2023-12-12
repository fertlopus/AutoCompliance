import argparse
import logging
from train import main as train_main
from inference import EmailPredictor
import sys


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Email Classification CLI")
    parser.add_argument('--train', action='store_true', help="Train the model")
    parser.add_argument('--predict', type=str, help="Classify a given text")
    args = parser.parse_args()

    try:
        if args.train:
            train_main()

        if args.predict:
            # TODO: Hardcoded path need to change
            predictor = EmailPredictor('./../model_artifacts/lr/email_classifier.pkl',
                                       "./../model_artifacts/lr/tfidf_vectorizer.pkl")
            prediction = predictor.predict(args.predict)
            logger.info(f"Predicted category: {prediction}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
