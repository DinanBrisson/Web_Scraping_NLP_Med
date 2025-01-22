import streamlit as st
import sys
import os
import pandas as pd
import re  # To clean file names

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Ranker.ranker import ArticleRanker
from deep_translator import GoogleTranslator


class MedicalArticleSearchApp:
    def __init__(self):
        """Initialize the ranking model."""
        self.ranker = ArticleRanker()

    @staticmethod
    def translate_to_english(word):
        """Translate a given word into English."""
        try:
            return GoogleTranslator(target="en").translate(word)
        except Exception as e:
            st.warning(f"Translation failed: {e}")
            return word

    @staticmethod
    def clean_filename(query):
        """Sanitize query to create a safe filename."""
        return re.sub(r'[^a-zA-Z0-9_-]', '_', query)

    @staticmethod
    def save_results_to_csv(ranked_articles, filename):
        """Save ranked articles to a CSV file."""
        os.makedirs("Data", exist_ok=True)  # Ensure Data directory exists
        df = pd.DataFrame(ranked_articles)
        df.to_csv(filename, index=False)
        print(f"\n[INFO] Results saved to {filename}")

    @staticmethod
    def convert_to_csv(data):
        """Convert ranked articles to CSV format for download."""
        df = pd.DataFrame(data)
        return df.to_csv(index=False).encode('utf-8')

    def run(self):
        """Run the Streamlit app."""
        st.title("Medical Article Search")

        # Add a description of the app
        st.write("""
        This application helps you **search and rank medical research articles related to kidney diseases**.  
        It uses **BioBERT**, a biomedical NLP model, to analyze and find the most relevant **PubMed** articles based on your input query.

        ### **How it Works**:
        1. **Enter a medical term** in the search bar below.
        2. The system will search for **kidney-related articles**.
        3. The most relevant articles will be displayed with their **titles, journals, and links**.
        4. **Download results as a CSV file**.
        """)

        query = st.text_input("Enter a term")

        if query:
            translated_query = self.translate_to_english(query)
            cleaned_query = self.clean_filename(translated_query)  # Clean the query for filename

            ranked_articles = self.ranker.rank_articles(translated_query)

            if ranked_articles:
                # Print results to console
                print("\n[INFO] Top 10 Ranked Articles:\n")
                for i, article in enumerate(ranked_articles):
                    print(f"[{i + 1}] Title: {article['title']}")
                    print(f"    Journal: {article.get('journal')}")
                    print(f"    URL: {article['url']}")
                    print(f"    Score: {article['score']}")
                    print("-" * 80)

                # Save results to CSV with query-based filename
                csv_filename = f"Data/ranked_articles_{cleaned_query}.csv"
                self.save_results_to_csv(ranked_articles, csv_filename)

                # Display articles in Streamlit
                for article in ranked_articles:
                    st.subheader(article["title"])
                    st.write(f"**Journal**: {article['journal']}")
                    st.write(f"**[Read More]({article['url']})**")
                    st.write("---")

                # Convert data to CSV and add a download button with query name
                csv_data = self.convert_to_csv(ranked_articles)
                st.download_button(
                    label="Download Results",
                    data=csv_data,
                    file_name=f"ranked_articles_{cleaned_query}.csv",
                    mime="text/csv"
                )

            else:
                st.warning("No matching articles found.")


if __name__ == "__main__":
    app = MedicalArticleSearchApp()
    app.run()
