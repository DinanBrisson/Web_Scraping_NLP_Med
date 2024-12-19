import re
import os
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Necessary to download nltk resources
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def check_and_download_nltk_resources():
    """
    Check if NLTK resources are available locally, and download them if not.

    Returns:
    None
    """
    nltk_paths = nltk.data.path
    nltk_data_found = False

    for path in nltk_paths:
        if os.path.exists(path):
            nltk_data_found = True
            print(f"nltk_data directory found: {path}")
            break

    if not nltk_data_found:
        print("Downloading all NLTK resources...")
        nltk.download()

# NLTK local path
nltk.data.path.append('nltk_data')


class TextCleaner:
    """
    A class for cleaning and preprocessing text data, including abstracts and authors.

    Attributes:
    stop_words (set): A set of English stopwords.
    lemmatizer (WordNetLemmatizer): A lemmatizer for normalizing words.
    contextual_stopwords_to_keep (set): A set of stopwords to retain for contextual relevance.
    """
    def __init__(self):
        """
        Initialize the TextCleaner with stopwords and a lemmatizer.
        """
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Stopwords to keep
        self.contextual_stopwords_to_keep = {'with', 'of', 'to', 'in', 'and', 'or', 'not', 'no'}

    def clean_text(self, text):
        """
        Clean and preprocess text data.

        Parameters:
        text (str): The raw text to clean.

        Returns:
        list: A list of cleaned and lemmatized tokens.
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove any text before and including ":"
        text = re.sub(r'^[^:]*:', '', text).strip()

        # Retain letters + basic punctuation
        text = re.sub(r'[^a-zA-Z\s\-.,()\'\"]', '', text)

        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Lowercase and tokenize
        tokens = nltk.word_tokenize(text.lower())

        # Remove stopwords but keep the contextual ones
        tokens = [
            word for word in tokens
            if word not in self.stop_words or word in self.contextual_stopwords_to_keep
        ]

        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        return text

    def clean_authors(self, authors):
        """
        Clean and format the authors field.

        Parameters:
        authors (str): The raw authors string.

        Returns:
        str: A cleaned and formatted string of unique authors.
        """
        if not isinstance(authors, str):
            return "No authors available"

        # Remove numbers, line breaks, and non-breaking spaces
        authors = re.sub(r'\d+|\n| ', '', authors)

        # Replace multiple commas with a single comma
        authors = re.sub(r',\s*,+', ',', authors)

        # Remove trailing commas and normalize spacing
        authors = re.sub(r',\s*$', '', authors).strip()

        # Remove duplicates and preserve order
        author_list = [author.strip() for author in authors.split(',') if author.strip()]
        unique_authors = list(dict.fromkeys(author_list))  # Removes duplicates while maintaining order

        # Rejoin cleaned authors
        return ', '.join(unique_authors)

    def preprocess(self, articles_data):
        """
        Preprocess a list of articles, cleaning authors while keeping the original text.

        Parameters:
        articles_data (list): A list of dictionaries, where each dictionary represents an article.

        Returns:
        list: A list of dictionaries with original text and cleaned metadata.
        """
        cleaned_articles = []
        for article in articles_data:
            # Keep the original abstract
            abstract = article.get("Abstract", "")

            # Clean authors field
            authors = article.get("Authors", "")
            authors = self.clean_authors(authors)

            cleaned_article = {
                "Title": article.get("Title"),
                "Authors": authors,
                "Year": article.get("Year"),
                "Journal": article.get("Journal"),
                "URL": article.get("URL"),
                "Keywords": article.get("Keywords"),
                "Original_Abstract": abstract
            }
            cleaned_articles.append(cleaned_article)

        return cleaned_articles

