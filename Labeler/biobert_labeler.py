import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

class BioBERTLabeler:
    """
    A class to extract named entities from biomedical text using the BioBERT model.

    Attributes:
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the BioBERT model.
    model : transformers.PreTrainedModel
        Pretrained BioBERT model for token classification.
    ner_pipeline : transformers.pipelines.Pipeline
        Named Entity Recognition (NER) pipeline initialized with the BioBERT model.
    data : pandas.DataFrame
        Data loaded from a CSV file, containing the text to analyze.
    text_column : str
        Name of the column containing text data for processing.
    """

    def __init__(self):
        """
        Initializes the BioBERT model and its associated tokenizer and pipeline for NER.
        """
        print("[INFO] Initializing BioBERT model for Named Entity Recognition...")
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        self.model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )
        print("[INFO] BioBERT initialization complete.")

    def load_data(self, file_path, text_column):
        """
        Loads data from a CSV file and filters rows with missing text in the specified column.

        Parameters:
        file_path : str
            Path to the CSV file containing the data.
        text_column : str
            Name of the column containing the text to process.
        """
        print("[INFO] Loading data from CSV file...")
        self.data = pd.read_csv(file_path)
        self.data = self.data.dropna(subset=[text_column])
        self.text_column = text_column
        print(f"[INFO] Loaded {len(self.data)} rows.")

    def extract_entities(self):
        """
        Extracts named entities from the text using the BioBERT NER pipeline.

        Truncates input text to 512 tokens to comply with model input size limitations and
        processes the text to extract unique entities.
        """
        print("[INFO] Extracting entities using BioBERT...")

        def extract_with_biobert(text):
            """
            Truncates the text, processes it with BioBERT, and extracts unique entities.

            Parameters:
            text : str
                Text to process.

            Returns:
            str
                A comma-separated string of unique entities extracted from the text.
            """
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            truncated_text = self.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
            results = self.ner_pipeline(truncated_text)
            entities = set(result['word'] for result in results)
            return ", ".join(entities)

        self.data["Labels"] = self.data[self.text_column].apply(extract_with_biobert)
        print("[INFO] BioBERT entity extraction complete.")

    def save_results(self, output_file):
        """
        Saves the processed DataFrame with extracted entities to a CSV file.

        Parameters:
        output_file : str
            Path to the output CSV file.
        """
        print("[INFO] Saving results to a CSV file...")
        self.data.to_csv(output_file, index=False)
        print(f"[INFO] Results saved to {output_file}.")
