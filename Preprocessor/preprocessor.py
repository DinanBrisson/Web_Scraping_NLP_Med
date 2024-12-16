import re
import os
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Necessary to download nltk ressources
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def check_and_download_nltk_resources():
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


# Nltk local path
nltk.data.path.append('nltk_data')


class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Stopwords to keep
        self.contextual_stopwords_to_keep = {'with', 'of', 'to', 'in', 'and', 'or', 'not', 'no'}

    def clean_text(self, text):
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

        return tokens

        # Rejoin cleaned sentence
        # return ' '.join(tokens)

    def clean_authors(self, authors):
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
        # Process each article and retain all metadata with the cleaned abstract
        cleaned_articles = []
        for article in articles_data:
            # Clean the abstract
            abstract = article.get("Abstract", "")
            if isinstance(abstract, str):  # Ensure abstract is a string
                cleaned_text = self.clean_text(abstract)
            else:
                cleaned_text = "Not available"

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
                "Cleaned_Abstract": cleaned_text
            }
            cleaned_articles.append(cleaned_article)

        return cleaned_articles
