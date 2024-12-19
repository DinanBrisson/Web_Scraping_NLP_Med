import spacy
import pandas as pd

class ScispacyLabeler:
    def __init__(self):
        print("[INFO] Initializing the SpaCy NLP model...")
        self.nlp = spacy.load("en_ner_bc5cdr_md")  # Load the SpaCy model
        self.data = None
        print("[INFO] Initialization complete.")

    def load_data(self, file_path):
        print("[INFO] Loading data from the CSV file...")
        self.data = pd.read_csv(file_path)  # Load the CSV file
        self.data = self.data.dropna(subset=["Original_Abstract"])  # Remove rows with missing text
        print(f"[INFO] Data loaded with {len(self.data)} rows.")

    def extract_entities(self):
        print("[INFO] Extracting named entities...")

        # Function to process text and extract unique entities
        def process_text(text):
            doc = self.nlp(text)  # Process the text
            entities = {ent.text for ent in doc.ents}  # Use a set to avoid duplicates
            return ", ".join(sorted(entities))  # Return entities as a comma-separated string

        # Apply entity extraction directly on the column Original_Abstract
        self.data["Labels"] = self.data["Original_Abstract"].apply(process_text)
        print("[INFO] Entity extraction and labeling complete.")

    def save_results(self, output_file):
        print("[INFO] Saving the results to a CSV file...")
        self.data.to_csv(output_file, index=False)
        print(f"[INFO] Results saved to {output_file}.")
