# Web Scraping and NLP in the Medical Field

This project implements a complete pipeline for Web Scraping, Data Processing, Ranking, and an Interactive Application to efficiently exploit medical publications.

📂 Code Structure

📂 Project_M2├── 📁 Data/ → Contains extracted and cleaned data├── 📁 Preprocessor/ → Data preprocessing scripts├── 📁 Labeler/ → Text annotation using BioBERT and SciSpaCy├── 📁 Ranker/ → Ranking of articles based on relevance├── 📁 App/ → User interface├── 📜 requirements.txt → Project dependencies├── 📜 README.md → Documentation├── 📜 main.py → Main pipeline script

🚀 1️⃣ Web Scraping

The Web Scraping module extracts medical publications from PubMed and other sources.

📌 Main File: scraper.py

🔹 Technologies Used:

requests for fetching web content

BeautifulSoup for HTML parsing

Selenium for handling dynamic websites

🔹 Features:

Extracts titles, abstracts, authors, DOI, publication date

Exports data to CSV

🔹 Run the Scraper:

python scraper.py

🛠️ 2️⃣ Data Processing

Once extracted, raw data must be cleaned and prepared for NLP analysis.

📌 Main Files: preprocessor.py, vectorizer.py

🔹 Preprocessing Steps:

Removal of HTML tags, special characters

Tokenization and Lemmatization

Stopword removal

Vectorization using TF-IDF

🔹 Technologies Used:

NLTK and spaCy for text processing

Pandas for data management

🔹 Run Preprocessing:

python preprocessor.py

🔖 3️⃣ Data Annotation

Two methods are used for medical text annotation:

📌 BioBERT File: biobert_labeler.py

Uses BioBERT for medical entity extraction

Implements an NER pipeline with transformers

📌 SciSpaCy File: scispacy_labeler.py

Uses SciSpaCy for biomedical concept identification

Extracts medical terms from abstracts

🔹 Run Annotation:

python biobert_labeler.py
python scispacy_labeler.py

📊 4️⃣ Ranking

The ranking module orders extracted articles based on their relevance.

📌 Main Files: encoder.py, ranker.py

🔹 Implemented Methods:

TF-IDF: Calculates word importance scores

BioBERT: Generates embeddings and measures similarity

LIME: Explains article relevance

🔹 Run Ranking:

python ranker.py

🖥️ 5️⃣ Application

A user interface was developed using Streamlit to interact with the results.

📌 Main File: app.py

🔹 Features:

Search and display retrieved articles

Show relevance scores

Interactive data exploration

🔹 Run the Application:

streamlit run app.py

🔥 Conclusion

This project provides a complete approach from scraping to interactive data exploitation. Future improvements include:

🔹 Optimizing ranking with more advanced models

🔹 Adding new medical sources

🔹 Deploying the application online

👨‍💻 Author

📌 Dinan Brisson🔗 GitHub📧 Contact: dinan.brisson@example.com

