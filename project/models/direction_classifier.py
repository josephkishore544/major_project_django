import torch
from torch import nn
import numpy as np 

# For multi class prediction, output features = number of classes
class DirectionClassifier(nn.Module) :
    def __init__(self,input_features = 768, output_features = 14) :
        super().__init__()
        self.classifer = nn.Sequential(
                            nn.Linear(input_features, 256),
                            nn.Linear(256, 128),
                            nn.Linear(128, 64),
                            nn.Linear(64, output_features)
                        )

    def forward(self, sentence_embedding) :
        return self.classifer(sentence_embedding)

    def load_model(self, path) :
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path,map_location = device)
        self.load_state_dict(ckpt,strict=True)
        self.classifer.to(device=device)

    def get_label(self,label_proba) :
        class_label_list = ['age+', 'age-', 'gender+', 'gender-', 'eye_open+', 'eye_open-', 'mouth_open+', 'mouth_open-', 'smile+', 'smile-', 'nose+', 'nose-', 'pitch+', 'pitch-']
        label = class_label_list[np.argmax(label_proba.cpu().detach().numpy())]
        attribute = label[:-1]
        effect = 'increase' if label[-1] == '+' else 'decrease'
        return attribute,effect