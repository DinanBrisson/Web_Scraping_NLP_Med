import json

from Labeler.biobert_labeler import BioBERTLabeler
from Labeler.scispacy_labeler import ScispacyLabeler
from Scraper.scraper import PubMedScraper, save_to_json, json_to_csv
from Preprocessor.preprocessor import TextCleaner, check_and_download_nltk_resources
from Preprocessor.vectorizer import TextVectorizer, save_to_csv

if __name__ == "__main__":

    """
    url = "https://pubmed.ncbi.nlm.nih.gov/?term=%28%22kidney+injury%22+OR+%22renal+toxicity%22+OR+%22nephrotoxicity%22%29+AND+%28%22drug+therapy%22+OR+medication+OR+pharmacotherapy+OR+%22nephrotoxic+drugs+»%29&filter=datesearch.y_10&filter=simsearch1.fha&filter=pubt.clinicaltrial&filter=pubt.randomizedcontrolledtrial&filter=lang.english&filter=hum_ani.humans"

    # Initialize scraper
    scraper = PubMedScraper(pages=115, delay=1)
    scraper.url = url
    articles_data = scraper.scrape_articles()
    save_to_json(articles_data, "pubmed_scrap.json")
    # json_to_csv("pubmed_scrap.json", "pubmed_scrap.csv")
    article  = [article["Abstract"] for article in articles_data]
    save_to_json(article, "Abstracts.json")

    # Check and download if necessary
    check_and_download_nltk_resources()
    """

    """
    Load JSON data
    with open("pubmed_scrap.json", "r", encoding="utf-8") as file:
        articles_data = json.load(file)

    # Initialize and apply Preprocessor
    preprocessor = TextCleaner()
    cleaned_data = preprocessor.preprocess(articles_data)

    cleaned_texts = [article["Original_Abstract"] for article in cleaned_data]

    save_to_json(cleaned_texts, "Cleaned_Abstracts.json")

    save_to_json(cleaned_data, "pubmed_Cleaned.json")

    json_to_csv("pubmed_Cleaned.json", "pubmed_Cleaned.csv")
    """

    """
    # Initialize Vectorizer and transform data
    vectorizer = TextVectorizer()
    bow_matrix = vectorizer.fit_transform_bow(cleaned_texts)
    tfidf_matrix = vectorizer.fit_transform_tfidf(cleaned_texts)

    print("Bag of Words matrix shape:", bow_matrix.shape)
    save_to_csv(bow_matrix, vectorizer.vectorizer_bow.get_feature_names_out(), "BOW.csv")
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    save_to_csv(tfidf_matrix, vectorizer.vectorizer_tfidf.get_feature_names_out(), "TF-IDF.csv")
    """

    """
    # Entity recognition with SciSpacy
    labeler = ScispacyLabeler()
    input_file = "pubmed_Cleaned.csv"
    output_file = "Labeled_Abstracts_SciSpacy.csv"
    text_column = "Original_Abstract"

    labeler.load_data(input_file, text_column)
    labeler.extract_entities()
    labeler.save_results(output_file)
    """

    # Entity recognition with BioBERT
    labeler = BioBERTLabeler()
    input_file = "Data/pubmed_Cleaned.csv"
    output_file = "Labeled_Abstracts_BioBERT.csv"
    text_column = "Original_Abstract"

    labeler.load_data(input_file, text_column)
    labeler.extract_entities()
    labeler.save_results(output_file)
