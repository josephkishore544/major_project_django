import torch
from torch.utils.data import Dataset

# This class extends torch.utils.data.Dataset
# It is a map-style dataset
# Pass the path to sentence_embeddings.pt and
# attribute_labels.pt files when instantiating
class TextEncoderDataset(Dataset) :
  def __init__(self,sentence_embeddings_path,attribute_labels_path) :
    # Load data from given paths
    try :
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.sentence_embeddings = torch.load(sentence_embeddings_path, map_location = device)
      self.attribute_labels = torch.load(attribute_labels_path, map_location = device)
    except Exception as e :
      print(e)
    print("TextEncoderDataset is created")
  
  def __len__(self) :
    # Should return the length of dataset
    return len(self.sentence_embeddings)

  def __getitem__(self,idx) :
    # Should return a data point at given index
    data_point = {"sentence_embedding_tensor" : self.sentence_embeddings[idx], "attribute_labels_tensor" : self.attribute_labels[idx]}
    return data_point