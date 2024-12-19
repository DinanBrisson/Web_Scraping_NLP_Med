import spacy
import pandas as pd

class ScispacyLabeler:
    """
    A class to extract named medical entities from text using the SciSpaCy model.

    Attributes:
    nlp : spacy.language.Language
        Loaded SpaCy model for named entity recognition.
    data : pandas.DataFrame
        Data loaded from a CSV file, containing the text to analyze.

    Methods:
    load_data(file_path):
        Loads a CSV file and filters rows without text.
    extract_entities():
        Extracts named entities from the text and adds them to the DataFrame.
    save_results(output_file):
        Saves the results with extracted entities to a CSV file.
    """

    def __init__(self):
        """
        Initializes the SciSpaCy NLP model for medical named entity extraction.
        """
        print("[INFO] Initializing the SpaCy NLP model...")
        self.nlp = spacy.load("en_ner_bc5cdr_md")  # Load the SpaCy model
        self.data = None
        print("[INFO] Initialization complete.")

    def load_data(self, file_path):
        """
        Loads data from a CSV file and filters rows with missing text.

        Parameters:
        file_path : str
            Path to the CSV file containing the data.

        Returns:
        None
        """
        print("[INFO] Loading data from the CSV file...")
        self.data = pd.read_csv(file_path)  # Load the CSV file
        self.data = self.data.dropna(subset=["Original_Abstract"])  # Remove rows with missing text
        print(f"[INFO] Data loaded with {len(self.data)} rows.")

    def extract_entities(self):
        """
        Extracts unique named entities from the text in the "Original_Abstract" column.

        Returns:
        None
        """
        print("[INFO] Extracting named entities...")

        def process_text(text):
            """
            Processes a text to extract named entities.

            Parameters:
            text : str
                Text to analyze.

            Returns:
            str
                A string containing the extracted entities, separated by commas.
            """
            doc = self.nlp(text)  # Process the text
            entities = {ent.text for ent in doc.ents}  # Use a set to avoid duplicates
            return ", ".join(sorted(entities))  # Return entities as a comma-separated string

        self.data["Labels"] = self.data["Original_Abstract"].apply(process_text)
        print("[INFO] Entity extraction and labeling complete.")

    def save_results(self, output_file):
        """
        Saves the results to a CSV file.

        Parameters:
        output_file : str
            Path to the output CSV file.

        Returns:
        None
        """
        print("[INFO] Saving the results to a CSV file...")
        self.data.to_csv(output_file, index=False)
        print(f"[INFO] Results saved to {output_file}.")
