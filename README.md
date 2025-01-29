# Web Scraping and NLP in the Medical Field

This project implements a complete pipeline for Web Scraping, Data Processing, Ranking, and an Interactive Application to efficiently exploit medical publications.


## 1️⃣ Web Scraping

The Web Scraping module extracts medical publications from PubMed.

## 2️⃣ Data Processing

Once extracted, raw data must be cleaned and prepared for NLP analysis.

### Preprocessing Steps:

Removal of HTML tags, special characters, redundancy.

## 4️⃣ Ranking

The ranking module orders extracted articles based on their relevance with a user query.

### Files: encoder.py, ranker.py

#### Methods:

- BioBERT: Generates embeddings.
- Torch : Mesure cosine similarity and sort by score.
- LIME: Explains article relevance with the most important words.

## 5️⃣ Application

A user interface was developed using Streamlit to interact with the results.

### Features:

- Search and display retrieved articles
- Show relevance scores
- Interactive data exploration

### Run the Application:

streamlit run App/app.py

## Author

- Dinan Brisson, ISEN
  Github : https://github.com/DinanBrisson
  dinan.brisson@isen-ouest.yncrea.fr

