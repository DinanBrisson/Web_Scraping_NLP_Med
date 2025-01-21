import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

class EncodeAbstracts:
    """
    A class to precompute embeddings for article abstracts using BioBERT
    and save them to a file for future use.
    """

    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", batch_size=16,
                 input_csv="Data/pubmed_Cleaned.csv", output_file="Data/abstracts_embeddings.pt"):
        """
        Initializes the embedding preprocessor.

        :param model_name: Name of the transformer model to use (default: BioBERT)
        :param batch_size: Number of abstracts processed per batch
        :param input_csv: Path to the CSV file containing abstracts
        :param output_file: Path to save the precomputed embeddings
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.input_csv = input_csv
        self.output_file = output_file
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load BioBERT model
        print("[INFO] Loading BioBERT model...")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def load_data(self):
        """
        Loads the dataset and extracts abstracts.
        """
        print("[INFO] Loading dataset...")
        df = pd.read_csv(self.input_csv)

        abstracts = df["Original_Abstract"].tolist()
        print(f"[INFO] Found {len(abstracts)} abstracts.")
        return abstracts

    def encode_and_save(self):
        """
        Encodes abstracts using BioBERT and saves the embeddings to a file.
        """
        # Load abstracts
        abstracts = self.load_data()

        # Encode abstracts in batches
        print("[INFO] Encoding abstracts...")
        abstracts_embeddings = self.model.encode(abstracts, convert_to_tensor=True, device=self.device, batch_size=self.batch_size)

        # Save embeddings to disk
        torch.save(abstracts_embeddings, self.output_file)
        print(f"[INFO] Embeddings saved to {self.output_file}")
