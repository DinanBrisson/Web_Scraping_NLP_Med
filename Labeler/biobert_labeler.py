import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

class BioBERTLabeler:
    def __init__(self):
        print("[INFO] Initializing BioBERT model for Named Entity Recognition...")

        # Load BioBERT
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        self.model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

        """
        # Load PubMedBERT
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        self.model = AutoModelForTokenClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        """

        # Initialize the pipeline
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"  # Aggregate subword tokens
        )
        print("[INFO] BioBERT initialization complete.")

    def load_data(self, file_path, text_column):
        print("[INFO] Loading data from CSV file...")
        self.data = pd.read_csv(file_path)  # Load the CSV file
        self.data = self.data.dropna(subset=[text_column])  # Remove rows with missing text
        self.text_column = text_column  # Store the column name for future reference
        print(f"[INFO] Loaded {len(self.data)} rows.")

    def extract_entities(self):
        print("[INFO] Extracting entities using BioBERT...")

        # Function to truncate and process text
        def extract_with_biobert(text):
            # Truncate text manually to 512 tokens
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            truncated_text = self.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

            # Apply the NER pipeline to the truncated text
            results = self.ner_pipeline(truncated_text)
            entities = set([result['word'] for result in results])  # Eliminate duplicates
            return ", ".join(entities)  # Return as a comma-separated string

        # Apply BioBERT extraction directly on the specified column
        self.data["Labels"] = self.data[self.text_column].apply(extract_with_biobert)
        print("[INFO] BioBERT entity extraction complete.")

    def save_results(self, output_file):
        print("[INFO] Saving results to a CSV file...")
        self.data.to_csv(output_file, index=False)
        print(f"[INFO] Results saved to {output_file}.")
