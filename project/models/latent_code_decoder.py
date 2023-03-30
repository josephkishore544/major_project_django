import torch
from torch import nn

# Class to load and use LatentCodeDecoder Model
# Its a simple fully connected neural network
class LatentCodeDecoder(nn.Module) :
  def __init__(self, input_features = 37, output_features = 9216) :
    super().__init__()
    self.decode_labels = nn.Sequential(
                        nn.Linear(input_features, 72),
                        nn.ReLU(),
                        nn.Linear(72, 144),
                        nn.ReLU(),
                        nn.Linear(144, 288),
                        nn.ReLU(),
                        nn.Linear(288, 576),
                        nn.ReLU(),
                        nn.Linear(576, 1152),
                        nn.ReLU(),
                        nn.Linear(1152, 2304),
                        nn.ReLU(),
                        nn.Linear(2304, 4608),
                        nn.ReLU(),
                        nn.Linear(4608, output_features),
                        nn.ReLU(),
                        nn.Linear(output_features, output_features)
                      )
  def forward(self,attribute_labels) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = self.decode_labels(attribute_labels)
    res = res.to(device=device)
    return res

  def load_model(self, path) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path,map_location = device)
    self.load_state_dict(ckpt,strict=True)
    self.decode_labels.to(device=device)