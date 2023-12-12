from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import EmailPredictor

app = FastAPI()


class EmailText(BaseModel):
    text: str


@app.post("/predict")
def predict_email_category(email: EmailText):
    # Load  model
    predictor = EmailPredictor('./model_artifacts/lr/email_classifier.pkl',
                               './model_artifacts/lr/tfidf_vectorizer.pkl')

    # Make a prediction
    prediction = predictor.predict(email.text)
    return {"prediction": prediction}
