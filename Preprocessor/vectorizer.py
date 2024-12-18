import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def save_to_csv(matrix, feature_names, filename):
    """
    Save a sparse matrix to a CSV file with feature names as columns.

    Parameters:
    matrix (scipy.sparse matrix): The sparse matrix to save.
    feature_names (list): List of feature names corresponding to the columns.
    filename (str): Path and name of the file to save the data.

    Returns:
    None
    """
    df = pd.DataFrame(matrix.toarray(), columns=feature_names)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


class TextVectorizer:
    """
    A class to vectorize text data using Bag-of-Words (BoW) and TF-IDF.

    Attributes:
    vectorizer_bow (CountVectorizer): Vectorizer for Bag-of-Words representation.
    vectorizer_tfidf (TfidfVectorizer): Vectorizer for TF-IDF representation.
    """
    def __init__(self):
        """
        Initialize the TextVectorizer with CountVectorizer and TfidfVectorizer.
        """
        self.vectorizer_bow = CountVectorizer()
        self.vectorizer_tfidf = TfidfVectorizer()

    def fit_transform_bow(self, texts):
        """
        Fit and transform the text data using Bag-of-Words vectorization.

        Parameters:
        texts (list of str): A list of text data to vectorize.

        Returns:
        scipy.sparse matrix: The Bag-of-Words vectorized representation of the input texts.
        """
        return self.vectorizer_bow.fit_transform(texts)

    def fit_transform_tfidf(self, texts):
        """
        Fit and transform the text data using TF-IDF vectorization.

        Parameters:
        texts (list of str): A list of text data to vectorize.

        Returns:
        scipy.sparse matrix: The TF-IDF vectorized representation of the input texts.
        """
        return self.vectorizer_tfidf.fit_transform(texts)
