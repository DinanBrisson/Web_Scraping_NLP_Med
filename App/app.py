import os
import re
import sys
import streamlit as st
from matplotlib import pyplot as plt

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Ranker.ranker import ArticleRanker
from deep_translator import GoogleTranslator


class App:
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
        4. **Each article will have an individual LIME explanation plot**.
        """)

        # Initialize session state for query
        if "query" not in st.session_state:
            st.session_state.query = ""

        query = st.text_input("Enter a medical term", value=st.session_state.query)

        if query:
            # Save query in session state
            st.session_state.query = query

            translated_query = self.translate_to_english(query)
            cleaned_query = self.clean_filename(translated_query)

            with st.spinner("Searching for relevant articles..."):
                ranked_articles = self.ranker.rank_articles(translated_query)

            if ranked_articles:
                st.success(f"Found {len(ranked_articles)} relevant articles")

                for index, article in enumerate(ranked_articles):
                    st.subheader(f"**{index+1}. {article['title']}**")
                    st.write(f"**Journal**: {article['journal']}")
                    st.write(f"**Similarity Score**: {article['score']:.4f}")
                    st.write(f"**Matching Query Words**: {', '.join(article['matching_query_words']) if article['matching_query_words'] else 'None'}")
                    st.write(f"**Matching Renal Keywords**: {', '.join(article['matching_renal_words']) if article['matching_renal_words'] else 'None'}")
                    st.write(f"**[Read More]({article['url']})**")

                    # Run LIME explanation for each article
                    st.subheader(f"LIME Explanation for Article {index+1}")
                    with st.spinner("Generating explanation..."):
                        lime_df = self.ranker.lime_explainer(translated_query, article)

                    if lime_df is not None:
                        # Sort word importance by ascending order
                        lime_df = lime_df.sort_values(by="Importance", ascending=True)

                        # Define colors for the bars: Green (positive impact), Red (negative impact)
                        colors = ["green" if x > 0 else "red" for x in lime_df["Importance"]]

                        fig, ax = plt.subplots(figsize=(8, 6))

                        ax.barh(lime_df["Word"], lime_df["Importance"], color=colors)
                        ax.set_title(f"LIME - Word Importance for Article {index+1}")
                        ax.set_xlabel("Importance Score")
                        ax.set_ylabel("Word")
                        st.pyplot(fig)

                    else:
                        st.warning(f"No LIME explanation available for article {index+1}")

                    st.write("---")

            else:
                st.warning("No matching articles found. Please try another term.")


if __name__ == "__main__":
    app = App()
    app.run()
