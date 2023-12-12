import gradio as gr
from src.inference import EmailPredictor


def classify_email(text):
    # Load  model (Ensure the model and vectorizer are correctly loaded)
    predictor = EmailPredictor('./model_artifacts/lr/email_classifier.pkl',
                               './model_artifacts/lr/tfidf_vectorizer.pkl')
    prediction = predictor.predict(text)
    return prediction


# Create a Gradio interface
interface = gr.Interface(
    fn=classify_email,
    inputs=gr.Textbox(lines=2, placeholder="Enter Email Text Here..."),
    outputs="text"
)

# Launch the app
interface.launch()
