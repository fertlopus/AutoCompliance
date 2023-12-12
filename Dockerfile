# Python runtime
FROM python:3.8-slim

# working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# port 8000 available to the world outside this container
EXPOSE 8000

#  environment variable
ENV MODEL_PATH=/model_artifacts/lr/email_classifier.pkl
ENV VECTORIZER_PATH=/model_artifacts/lr/tfidf_vectorizer.pkl

# FastAPI server when the container launches
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
