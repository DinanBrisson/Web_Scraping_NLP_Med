import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoModel, AutoTokenizer

# Renal-related keywords with a weight factor
renal_keywords = {word.lower(): 3 for word in [
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
]}


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)  # Attention scores
        self.softmax = nn.Softmax(dim=1)  # Normalization
        self.fc = nn.Linear(hidden_size, hidden_size)  # Extra transformation
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # Normalize activations
        self.dropout = nn.Dropout(p=0.3)  # Dropout for regularization

    def forward(self, hidden_states, attention_mask, input_tokens):
        """
        Applies attention with a boost for renal-related keywords.
        """
        scores = self.attention(hidden_states).squeeze(-1)  # Compute attention scores

        # Boost attention scores for renal keywords
        for i, token_list in enumerate(input_tokens):
            for j, token in enumerate(token_list):
                token = token.lower()
                if token in renal_keywords:
                    scores[i, j] *= renal_keywords[token]  # Increase score

        scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        weights = self.softmax(scores)

        # Compute weighted sum of hidden states
        context_vector = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)

        # Apply batch normalization and dropout
        context_vector = self.batch_norm(context_vector)
        context_vector = self.dropout(context_vector)

        # Apply residual connection
        context_vector = self.fc(context_vector) + context_vector  # Skip connection
        context_vector = self.activation(context_vector)

        return context_vector


class EncodeAbstracts:
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", batch_size=16,
                 input_csv="Data/pubmed_Cleaned.csv", output_file="Data/abstracts_embeddings_attention_3.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.biobert = AutoModel.from_pretrained(model_name).to(self.device)
        self.attention = AttentionLayer(hidden_size=768).to(self.device)

        # Fully Connected Layer for Dimensionality Reduction (768 → 512)
        self.fc = nn.Linear(768, 512).to(self.device)
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(512)  # Normalization after reduction
        self.dropout = nn.Dropout(p=0.3)  # Dropout for regularization

        self.batch_size = batch_size
        self.input_csv = input_csv
        self.output_file = output_file

    def encode_abstracts(self):
        df = pd.read_csv(self.input_csv)
        abstracts = df["Original_Abstract"].tolist()
        all_embeddings = []

        for i in range(0, len(abstracts), self.batch_size):
            batch = abstracts[i:i + self.batch_size]

            # Tokenization with token retrieval
            tokens = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)
            input_tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

            # Pass through BioBERT
            with torch.no_grad():
                outputs = self.biobert(input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state

                # Apply Attention Mechanism
                context_vector = self.attention(hidden_states, attention_mask, input_tokens)

                # Apply Dimensionality Reduction (768 → 512)
                reduced_embeddings = self.fc(context_vector)
                reduced_embeddings = self.activation(reduced_embeddings)
                reduced_embeddings = self.batch_norm(reduced_embeddings)
                reduced_embeddings = self.dropout(reduced_embeddings)

            all_embeddings.append(reduced_embeddings.cpu())

        # Save reduced embeddings
        final_embeddings = torch.cat(all_embeddings)
        torch.save(final_embeddings, self.output_file)
        print(f"[INFO] Reduced embeddings saved: {self.output_file}")
