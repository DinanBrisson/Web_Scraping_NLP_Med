import torch
import pandas as pd
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, models, util


class ArticleRanker:
    """
    A class to rank PubMed articles based on their relevance to a given search term.
    It utilizes BioBERT to compute semantic similarity between the user's input and article abstracts.
    """

    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", embeddings_file="Data/abstracts_embeddings.pt",
                 input_csv="Data/pubmed_Cleaned.csv"):
        """
        Initializes the BioBERT model for text similarity evaluation.

        :param model_name: Pre-trained BioBERT model name from Hugging Face.
        :param embeddings_file: Path to precomputed embeddings file.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings_file = embeddings_file
        self.input_csv = input_csv
        self.articles = []
        self.embeddings = None

        # Load BioBERT model
        print("[INFO] Loading BioBERT model...")
        self.model = self.load_biobert(model_name)

        # Load articles and Precomputed embeddings
        self.load_articles()
        self.load_precomputed_embeddings()

    def load_biobert(self, model_name):
        """
        Loads the BioBERT model in a SentenceTransformer-compatible format.
        """
        word_embedding_model = models.Transformer(model_name)  # Load BioBERT model
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True)

        return SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def load_articles(self):
        """
        Loads articles from the CSV file and stores them in self.articles.
        """
        print("[INFO] Loading articles from CSV...")
        df = pd.read_csv(self.input_csv)

        # Rename columns and convert to list of dictionaries
        self.articles = df.rename(columns={"Title": "title", "Journal": "journal", "Original_Abstract": "abstract", "URL": "url"}).to_dict(
            orient="records")

        print(f"[INFO] Loaded {len(self.articles)} articles.")

    def load_precomputed_embeddings(self):
        """
        Loads precomputed embeddings from a file.
        """
        print("[INFO] Loading precomputed embeddings")
        try:
            self.embeddings = torch.load(self.embeddings_file)
            print(f"[INFO] Loaded precomputed embeddings: {self.embeddings.shape}")
        except FileNotFoundError:
            print("[ERROR] No precomputed embeddings found")
            exit()

    def rank_articles(self, user_input, top_n=10):
        """
        Ranks articles based on their relevance to the given input term.

        :param user_input: Search term or phrase provided by the user.
        :param top_n: Number of top-ranked articles to return.
        :return: List of top-ranked articles sorted by similarity score.
        """
        if not self.articles or self.embeddings is None:
            print("[INFO] No articles or embeddings available for analysis.")
            return []

        # Encode the user input
        print("\n[INFO] Encoding user query...")
        query_embedding = self.model.encode(user_input, convert_to_tensor=True, device=self.device)

        # Compute cosine similarity with precomputed embeddings
        print("[INFO] Calculating similarity scores...")
        similarities = util.pytorch_cos_sim(query_embedding, self.embeddings)

        # Sort articles by similarity score in descending order
        sorted_indices = torch.argsort(similarities, descending=True)

        # Retrieve the top-ranked articles
        ranked_articles = []
        for idx in sorted_indices[0][:top_n]:
            ranked_articles.append({
                "title": self.articles[idx]["title"],
                "abstract": self.articles[idx]["abstract"],
                "journal": self.articles[idx]["journal"],
                "url": self.articles[idx]["url"],
                "score": similarities[0, idx].item()
            })

        return ranked_articles

    def save_results_to_csv(self, ranked_articles, filename="ranked_articles.csv"):
        """
        Saves the ranked articles to a CSV file.
        """
        df = pd.DataFrame(ranked_articles)
        df.to_csv(filename, index=False)
        print(f"\n[INFO] Results saved to {filename}")

    def run(self):
        """
        Runs the user interaction: prompts the user for a search term, ranks articles, and saves the results.
        """
        user_input = input("Enter a search term: ")
        ranked_articles = self.rank_articles(user_input)

        if ranked_articles:
            print("\n[INFO] Top 10 Ranked Articles:\n")
            for i, article in enumerate(ranked_articles):
                print(f"[{i + 1}] Title: {article['title']}")
                print(f"    Journal: {article.get('journal')}")
                print(f"    URL: {article['url']}")
                print(f"    Score: {article['score']:.4f}")
                print("-" * 80)
            self.save_results_to_csv(ranked_articles)
        else:
            print("\n[INFO] No matching articles found.")
