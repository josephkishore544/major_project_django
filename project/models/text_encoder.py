import torch
from torch import nn

# CLass to load and use TextEncoder Model
# Its a simple fully connected neural network
class TextEncoder(nn.Module) :
  def __init__(self, input_features = 768, output_features = 37) :
    super().__init__()
    self.encode_text = nn.Sequential(
                        nn.Linear(input_features,384),
                        nn.ReLU(),
                        nn.Linear(384,192),
                        nn.ReLU(),
                        nn.Linear(192,48),
                        nn.ReLU(),
                        nn.Linear(48,output_features)
                      )
  def forward(self,sentence_embedding) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = self.encode_text(sentence_embedding)
    res = res.to(device=device)
    return res

  def load_model(self, path) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path,map_location = device)
    self.load_state_dict(ckpt,strict=True)
    self.encode_text.to(device=device)