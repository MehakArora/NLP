

"""
This script has functions for getting BERT embeddings for the text data.
"""

import torch
from transformers import BertTokenizer, BertModel
from typing import List, Tuple
import numpy as np

class BertEmbedder:
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get BERT embeddings for a list of texts.

        Args:
            texts (List[str]): List of text strings to get embeddings for.

        Returns:
            np.ndarray: Array of embeddings.
        """
        # Tokenize the input texts
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # Get the embeddings from the BERT model
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # The embeddings are in the last hidden state
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        
        return embeddings

# Example usage
if __name__ == "__main__":
    texts = ["Hello, how are you?", "I am fine, thank you!"]
    embedder = BertEmbedder()
    embeddings = embedder.get_embeddings(texts)
    import pdb; pdb.set_trace()
    print(embeddings)