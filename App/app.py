import os
import re
import sys
import pandas as pd
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
        4. **Download the articles as a CSV file**.
        5. **You can choose an article to have an individual LIME explanation**.
        """)

        # Initialize session state for query if not exists
        if "query" not in st.session_state:
            st.session_state.query = ""

        if "search_triggered" not in st.session_state:
            st.session_state.search_triggered = False

        # Input field for search query
        query = st.text_input("Enter a medical term", value=st.session_state.query)

        # Search button to submit query
        if st.button("Search 🔍"):
            if query != st.session_state.query:
                st.session_state.query = query  # Store new query
                st.session_state.search_triggered = True
                st.rerun()  # Rerun to update the displayed results

        # Ensure we only proceed if a search has been triggered
        if st.session_state.search_triggered and st.session_state.query:
            translated_query = self.translate_to_english(st.session_state.query)

            with st.spinner("Searching for relevant articles..."):
                ranked_articles = self.ranker.rank_articles(translated_query)

            if ranked_articles:
                st.success(f"Found {len(ranked_articles)} relevant articles")

                # Convert articles to DataFrame for CSV export
                articles_df = pd.DataFrame(ranked_articles)[["title", "journal", "score", "url"]]

                # Display ALL articles first
                for index, article in enumerate(ranked_articles):
                    st.subheader(f"**{index + 1}. {article['title']}**")
                    st.write(f"**Journal**: {article['journal']}")
                    st.write(f"**Similarity Score**: {article['score']:.4f}")
                    st.write(f"**Matching Query Words**: {', '.join(article['matching_query_words']) if article['matching_query_words'] else 'None'}")
                    st.write(f"**Matching Renal Keywords**: {', '.join(article['matching_renal_words']) if article['matching_renal_words'] else 'None'}")
                    st.write(f"**[Read More]({article['url']})**")
                    st.write("---")

                # Add a download button for articles CSV
                st.download_button(
                    label="Download Articles",
                    data=articles_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"Articles_{st.session_state.query.replace(' ', '_')}.csv",
                    mime="text/csv",
                )

                # Dropdown to select an article for LIME explanation
                st.subheader("Select an article for LIME Explanation")
                article_titles = [f"{index + 1}. {article['title']}" for index, article in enumerate(ranked_articles)]

                if article_titles:
                    selected_article_index = st.selectbox("Choose an article:", range(len(ranked_articles)),
                                                          format_func=lambda x: article_titles[x])
                    selected_article = ranked_articles[selected_article_index]

                    # Display selected article details
                    st.subheader(f"Selected Article: {selected_article['title']}")
                    st.write(f"**Journal**: {selected_article['journal']}")
                    st.write(f"**Similarity Score**: {selected_article['score']:.4f}")
                    st.write(f"**[Read More]({selected_article['url']})**")

                    # Generate LIME explanation **only after selection**
                    if st.button("Generate LIME Explanation"):
                        with st.spinner("Generating explanation..."):
                            lime_df = self.ranker.lime_explainer(translated_query, selected_article)

                        if lime_df is not None:
                            lime_df = lime_df.sort_values(by="Importance", ascending=True)

                            # Define colors for word importance impact
                            colors = ["green" if x > 0 else "red" for x in lime_df["Importance"]]

                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.barh(lime_df["Word"], lime_df["Importance"], color=colors)
                            ax.set_title(f"LIME - Word Importance for Selected Article")
                            ax.set_xlabel("Importance Score")
                            ax.set_ylabel("Word")
                            st.pyplot(fig)
                        else:
                            st.warning("No LIME explanation available for this article.")
                else:
                    st.warning("No articles available for explanation. Please try a different search term.")

            else:
                st.warning("No matching articles found. Please try another term.")


if __name__ == "__main__":
    app = App()
    app.run()
