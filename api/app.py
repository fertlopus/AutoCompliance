from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import EmailPredictor
import logging
from datetime import datetime

app = FastAPI()

# Setting up the basic logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


class EmailText(BaseModel):
    text: str


@app.post("/predict")
def predict_email_category(email: EmailText):
    try:
        start_time = datetime.now()
        # Load  model
        predictor = EmailPredictor('./model_artifacts/lr/email_classifier.pkl',
                               './model_artifacts/lr/tfidf_vectorizer.pkl')
        # Make a prediction
        prediction = predictor.predict(email.text)
        end_time = datetime.now()

        # Log the request and response details
        logger.info(f"Request received at {start_time}, Response sent at {end_time}, Prediction: {prediction}")
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code = 500, detail = "Internal Server Error")
