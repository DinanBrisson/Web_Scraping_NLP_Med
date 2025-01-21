import pandas as pd
import spacy


class SpacyFilter:
    """
    A class to perform pre-filtering of abstracts based on semantic similarity
    between extracted labels and a list of terms or phrases provided by the user.

    Attributes:
        similarity_threshold (float): Minimum similarity score to consider a match.
        nlp (spacy.lang): SpaCy model used for vector similarity calculations.
        data (pd.DataFrame): Data loaded from a CSV file.
    """

    def __init__(self, similarity_threshold):
        """
        Initializes the SpaCy filter for exact or close matches.

        Args:
            similarity_threshold (float): Minimum similarity score to consider a match (0.0-1.0).
        """
        print("[INFO] Initializing SpaCy for exact matching and filtering...")
        self.nlp = spacy.load('en_core_web_md')  # Load SpaCy model
        self.similarity_threshold = similarity_threshold  # Set similarity threshold
        self.data = None  # Placeholder for loaded data
        print("[INFO] SpaCy initialization complete.")

    def load_data(self, file_path):
        """
        Loads the labeled data from a CSV file and performs basic cleaning.

        Args:
            file_path (str): Path to the CSV file.

        Raises:
            ValueError: If the CSV file does not contain a 'Labels' column.
        """
        print("[INFO] Loading data from CSV file...")
        self.data = pd.read_csv(file_path)  # Load data
        # Check if the 'Labels' column exists in the data
        if "Labels" not in self.data.columns:
            raise ValueError("[ERROR] The CSV file must contain a 'Labels' column.")

        # Clean the data: Remove empty or invalid labels
        self.data.dropna(subset=["Labels"], inplace=True)  # Remove rows with empty labels
        self.data = self.data[self.data["Labels"].str.strip() != ""]  # Remove rows with empty strings
        print(f"[INFO] Loaded and cleaned {len(self.data)} rows.")

    def get_user_input(self):
        """
        Allows the user to input terms or phrases to search for.

        Returns:
            list: List of terms or phrases entered by the user.
        """
        print("[INFO] Please enter the terms you want to search for, separated by commas.")
        # Prompt the user to input terms separated by commas
        user_input = input("Enter terms: ")
        return [term.strip() for term in user_input.split(",")]  # Clean and return the terms

    def pre_filter(self, user_input):
        """
        Filters abstracts based on labels equivalent to or very close to the user's input.

        Args:
            user_input (list): List of terms or phrases provided by the user.

        Returns:
            pd.DataFrame: Subset of data containing labels equivalent or very close to the user's input.
        """
        print("[INFO] Starting pre-filtering process...")

        # Create SpaCy vector objects for each user input term
        user_vectors = [self.nlp(term) for term in user_input]

        def is_equivalent(label):
            """
            Checks if a label is equivalent (or close) to a user-provided term.

            Args:
                label (str): A label to compare.

            Returns:
                bool: True if the label is equivalent to a user term, False otherwise.
            """
            if not label.strip():  # Ignore empty labels
                return False

            label_vector = self.nlp(label)  # Convert the label to a SpaCy vector
            if not label_vector.has_vector:  # Skip if the vector is empty
                return False

            # Compare similarity with each user-provided term
            for user_vector in user_vectors:
                if user_vector.has_vector and label_vector.similarity(user_vector) >= self.similarity_threshold:
                    return True  # Match found
            return False  # No match found

        # Filter abstracts based on labels
        filtered_data = self.data[self.data["Labels"].apply(
            lambda labels: any(is_equivalent(label.strip()) for label in labels.split(",") if label.strip())
        )]
        print(f"[INFO] Filtering complete. {len(filtered_data)} rows retained.")
        return filtered_data

    def save_results(self, filtered_data, output_file):
        """
        Saves the filtered results to a CSV file.

        Args:
            filtered_data (pd.DataFrame): Filtered data.
            output_file (str): Path to the output file.
        """
        print("[INFO] Saving filtered results to CSV file...")
        filtered_data.to_csv(output_file, index=False)  # Save the data to a CSV file
        print(f"[INFO] Results saved to {output_file}.")
