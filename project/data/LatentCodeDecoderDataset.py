import torch
from torch.utils.data import Dataset

# This class extends torch.utils.data.Dataset
# It is a map-style dataset
# Pass the path to attribute_labels.pt and
# wplus_vectors.pt files when instantiating
class LatentCodeDecoderDataset(Dataset) :
  def __init__(self,attribute_labels_path,wplus_vectors_path) :
    # Load data from given paths
    try :
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.attribute_labels = torch.load(attribute_labels_path, map_location = device)
      self.wplus_vectors = torch.load(wplus_vectors_path, map_location = device)
    except Exception as e :
      print(e)
    print("LatentCodeDecoderDataset is created")
  
  def __len__(self) :
    # Should return the length of dataset
    return len(self.wplus_vectors)

  def __getitem__(self,idx) :
    # Should return a data point at given index
    idx_ = idx*10
    wplus = self.wplus_vectors[idx][0]
    reshaped_wplus = torch.reshape(wplus, (-1,))
    data_point = {"attribute_labels_tensor" : self.attribute_labels[idx_], "wplus_vector_tensor" : reshaped_wplus}
    return data_point