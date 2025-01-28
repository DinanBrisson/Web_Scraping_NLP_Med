import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer, models, util
from deep_translator import GoogleTranslator
from lime.lime_text import LimeTextExplainer


def translate_to_english(word):
    """
    Translates a word to English using Deep Translator with automatic language detection.
    """
    try:
        translated = GoogleTranslator(target="en").translate(word)  # Auto-detect language
        return translated
    except Exception as e:
        print(f"[WARNING] Translation failed: {e}")
        return word  # Return original word if translation fails


def visualize_word_importance(df):
    """
    Plots a bar chart of the most important words for classification.
    """
    plt.figure(figsize=(10, 6))
    plt.barh(df["Word"], df["Importance"], color=["green" if x > 0 else "red" for x in df["Importance"]])
    plt.xlabel("Importance Score")
    plt.ylabel("Word")
    plt.title("LIME - Word Importance")
    plt.gca().invert_yaxis()  # Flip the order so the most important word is on top
    plt.show()


class ArticleRanker:
    """
    A class to rank PubMed articles based on their relevance to a given search term.
    It utilizes BioBERT to compute semantic similarity between the user's input and article abstracts.
    """

    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", embeddings_file="Data/abstracts_embeddings.pt",
                 input_csv="Data/pubmed_Cleaned.csv"):
        """
        Initializes the BioBERT model for text similarity evaluation.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings_file = embeddings_file
        self.input_csv = input_csv
        self.articles = []
        self.embeddings = None

        # Load BioBERT model
        print("[INFO] Loading BioBERT model...")
        self.model = self.load_biobert(model_name)

        # Load articles and precomputed embeddings
        self.load_articles()
        self.load_precomputed_embeddings()

    @staticmethod
    def load_biobert(model_name):
        """
        Loads the BioBERT model in a SentenceTransformer-compatible format.
        """
        word_embedding_model = models.Transformer(model_name)
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
        self.articles = df.rename(columns={"Title": "title", "Journal": "journal",
                                           "Original_Abstract": "abstract", "URL": "url"}).to_dict(orient="records")

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

    def rank_articles(self, translated_query, top_n=10):
        """
        Ranks articles based on their relevance to the given input term.
        """
        if not self.articles or self.embeddings is None:
            print("[INFO] No articles or embeddings available for analysis.")
            return []

        print("\n[INFO] Encoding user query...")
        query_embedding = self.model.encode(translated_query, convert_to_tensor=True, device=self.device)

        print("[INFO] Calculating similarity scores...")
        similarities = util.pytorch_cos_sim(query_embedding, self.embeddings)

        # Sort articles by similarity score in descending order
        sorted_indices = torch.argsort(similarities, descending=True)

        renal_keywords = [
            # Renal diseases and pathologies
            "acute kidney injury", "AKI", "chronic kidney disease", "CKD",
            "end-stage renal disease", "ESRD", "nephrotic syndrome", "nephritis",
            "glomerulonephritis", "interstitial nephritis", "pyelonephritis",
            "diabetic nephropathy", "hypertensive nephropathy", "lupus nephritis",
            "focal segmental glomerulosclerosis", "FSGS", "polycystic kidney disease",
            "PKD", "renal cell carcinoma", "RCC", "urolithiasis", "nephrolithiasis",
            "kidney stones", "urinary tract infection", "UTI", "nephrolithiasis",
            "medullary cystic kidney disease", "IgA nephropathy", "membranous nephropathy",
            "thrombotic microangiopathy", "amyloidosis", "Alport syndrome", "Fabry disease",

            # Symptoms and complications
            "proteinuria", "hematuria", "albuminuria", "oliguria", "anuria",
            "azotemia", "hyperkalemia", "hypokalemia", "hypernatremia", "hyponatremia",
            "metabolic acidosis", "respiratory acidosis", "respiratory alkalosis",
            "fluid overload", "electrolyte imbalance", "uremia", "hypertension",
            "nephrogenic diabetes insipidus", "hypoalbuminemia", "hyperphosphatemia",
            "hypophosphatemia", "hypocalcemia", "hypercalcemia", "hypomagnesemia",
            "hypermagnesemia", "hyperparathyroidism", "osteodystrophy", "anemia of CKD",

            # Renal failure and treatments
            "renal insufficiency", "acute renal failure", "chronic renal failure",
            "dialysis", "hemodialysis", "peritoneal dialysis", "continuous renal replacement therapy",
            "CRRT", "extracorporeal dialysis", "kidney transplant", "renal replacement therapy",
            "RRT", "transplant rejection", "immunosuppressive therapy", "plasma exchange",
            "glomerular hyperfiltration", "refractory nephrotic syndrome", "steroid-resistant nephrotic syndrome",

            # Nephrotoxicity and drug-induced kidney injuries
            "nephrotoxicity", "drug-induced nephrotoxicity", "contrast-induced nephropathy",
            "CIN", "NSAID-induced nephropathy", "aminoglycoside nephrotoxicity",
            "vancomycin nephrotoxicity", "ACE inhibitor nephrotoxicity",
            "cisplatin nephrotoxicity", "radiocontrast nephropathy",
            "acute tubular necrosis", "ATN", "ischemic nephropathy",
            "cyclosporine nephrotoxicity", "tacrolimus nephrotoxicity",

            # General renal-related medical terms
            "renal failure", "kidney dysfunction", "glomerular filtration rate",
            "GFR", "creatinine clearance", "eGFR", "blood urea nitrogen", "BUN",
            "hydronephrosis", "hyperphosphatemia", "hypocalcemia", "hypercalcemia",
            "hypomagnesemia", "renal osteodystrophy", "nephritic syndrome",
            "nephrotic syndrome", "nephrocalcinosis", "renal fibrosis",
            "glomerular hypertrophy", "tubulointerstitial nephritis"
        ]

        ranked_articles = []
        for idx in sorted_indices[0]:
            article = self.articles[idx]
            abstract = article["abstract"].lower()

            # Find matching words from the translated query
            query_words = set(translated_query.lower().split())
            matching_query_words = [word for word in query_words if word in abstract]

            # Find matching renal-related words
            matching_renal_words = [kw for kw in renal_keywords if kw in abstract]

            # Only add articles that match both the query and renal-related words
            if matching_query_words and matching_renal_words:
                ranked_articles.append({
                    "title": article["title"],
                    "abstract": article["abstract"],
                    "journal": article["journal"],
                    "url": article["url"],
                    "score": similarities[0, idx].item(),
                    "matching_query_words": matching_query_words,
                    "num_matching_query_words": len(matching_query_words),
                    "matching_renal_words": matching_renal_words,
                    "num_matching_renal_words": len(matching_renal_words)
                })

            if len(ranked_articles) >= top_n:
                break

        return ranked_articles

    def explain_top1_article_with_lime(self, query, top_article):
        """
        Uses LIME to explain why the top-ranked article was selected.
        Identifies which words in the abstract contribute most to the ranking.
        """
        if not top_article or self.embeddings is None:
            print("[INFO] No ranked article available for LIME analysis.")
            return

        explainer = LimeTextExplainer(class_names=["Not Relevant", "Relevant"])

        def predict_proba(texts):
            """
            Computes similarity scores for LIME.
            Returns continuous probability scores instead of binary labels.
            """
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(embeddings, self.embeddings).cpu().numpy()

            # Normalize scores to probability range (0 to 1)
            min_score = similarities.min()
            max_score = similarities.max()

            if max_score - min_score == 0:  # Prevent division by zero
                normalized_scores = np.ones((len(similarities), 1)) * 0.5
            else:
                normalized_scores = (similarities - min_score) / (max_score - min_score)

            return np.hstack([1 - normalized_scores, normalized_scores])  # Convert to probability

        abstract = top_article["abstract"]

        print(f"\n[INFO] Explaining ranking for the top-ranked article: {top_article['title']}")
        print(f"    Score: {top_article['score']:.4f}")
        print(f"    URL: {top_article['url']}")

        # Generate LIME explanation
        explanation = explainer.explain_instance(
            abstract,
            predict_proba,
            num_features=20,
            num_samples=100,
            top_labels=1
        )

        # Extract words & their importance scores
        top_label = explanation.top_labels[0]  # Get the top label assigned by LIME
        importance_scores = explanation.as_list(label=top_label)  # Get feature importance

        print("\n[INFO] Words contributing to ranking:")
        for word, score in importance_scores:
            print(f"  {word}: {score:.4f}")

        # Save word importance to CSV
        df = pd.DataFrame(importance_scores, columns=["Word", "Importance"])
        df.to_csv("lime_word_importance.csv", index=False)
        print("\n[INFO] Word importance scores saved to 'lime_word_importance.csv'.")

        # 🔹 Step 3: Visualize Word Importance
        visualize_word_importance(df)

    def run(self):
        """
        Runs the user interaction: prompts the user for a search term, ranks articles, and saves the results.
        """
        user_input = input("Enter a search term: ")

        # Translate query to English
        translated_query = translate_to_english(user_input)
        print(f"[INFO] Using translated query: {translated_query}")

        ranked_articles = self.rank_articles(translated_query)

        if ranked_articles:
            print("\n[INFO] Top 10 Ranked Articles:\n")
            for i, article in enumerate(ranked_articles):
                print(f"[{i + 1}] Title: {article['title']}")
                print(f"    Journal: {article['journal']}")
                print(f"    Link: {article['url']}")
                print(f"    Score: {article['score']:.4f}")
                print(f"    Matching Query Words: {article['matching_query_words']}")
                print(f"    Matching Renal Words: {article['matching_renal_words']}")
                print("-" * 80)

            # Run LIME analysis on the top-ranked article only
            print("\n[INFO] Running LIME analysis for the top-ranked article...\n")
            self.explain_top1_article_with_lime(translated_query, ranked_articles[0])

        else:
            print("\n[INFO] No matching articles found.")
