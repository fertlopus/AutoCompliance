import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from typing import List, Tuple


def feature_extraction(texts: pd.Series, max_features: int = 1000) -> Tuple[TfidfVectorizer, pd.DataFrame]:
    """
    Perform TF-IDF feature extraction on the dataset.

    Args:
    texts (pd.Series): The text data for feature extraction.
    max_features (int): The maximum number of features to extract.

    Returns:
    Tuple[TfidfVectorizer, pd.DataFrame]: The fitted TF-IDF vectorizer and the transformed data as a DataFrame.
    """
    tfidf_vectorizer = TfidfVectorizer(max_features = max_features)
    features = tfidf_vectorizer.fit_transform(texts)

    # Convert to DataFrame for easier handling
    features_df = pd.DataFrame(features.toarray(), columns = tfidf_vectorizer.get_feature_names_out())

    return tfidf_vectorizer, features_df


def inverse_transform_feature(vectorizer: TfidfVectorizer, feature_vector: pd.Series) -> List[str]:
    """
    Inverse transform the feature vector to get the corresponding text data.

    Args:
    vectorizer (TfidfVectorizer): The TF-IDF vectorizer used for transforming the text.
    feature_vector (pd.Series): The feature vector to be inverse transformed.

    Returns:
    List[str]: The text data corresponding to the feature vector.
    """
    return vectorizer.inverse_transform(feature_vector.values.reshape(1, -1))[0]


def train_word2vec_model(texts: List[str], vector_size: int = 100, window: int = 5, min_count: int = 2) -> Word2Vec:
    """
    Train a Word2Vec model on the given texts.

    Args:
    texts (List[str]): A list of preprocessed texts (tokenized).
    vector_size (int): Dimensionality of the word vectors.
    window (int): Maximum distance between the current and predicted word within a sentence.
    min_count (int): Ignores all words with total frequency lower than this.

    Returns:
    Word2Vec: The trained Word2Vec model.
    """
    tokenized_texts = [text.split() for text in texts]  # Simple tokenization, adjust as needed
    model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=window, min_count=min_count)
    return model


def generate_embeddings(model: Word2Vec, tokenized_text: List[str]) -> List[float]:
    """
    Generate embeddings for a given tokenized text using a trained Word2Vec model.

    Args:
    model (Word2Vec): A trained Word2Vec model.
    tokenized_text (List[str]): A tokenized text (list of words).

    Returns:
    List[float]: The averaged word vector representation of the text.
    """
    embeddings = [model.wv[word] for word in tokenized_text if word in model.wv]
    if embeddings:
        return list(np.mean(embeddings, axis=0))
    else:
        return list(np.zeros(model.vector_size))
