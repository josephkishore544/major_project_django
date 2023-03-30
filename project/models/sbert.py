from sentence_transformers import SentenceTransformer
import torch

# Class to load and use fine-tuned SBERT
class sbert :
    def __init__(self, model_path) :
        self.sbert_model = SentenceTransformer(model_path)
    
    def encode(self, text) :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sentence_embedding_tensor = self.sbert_model.encode(text, convert_to_tensor = True)
        sentence_embedding_tensor = sentence_embedding_tensor.to(device=device)
        return sentence_embedding_tensor