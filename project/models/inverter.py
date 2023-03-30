from project.models.encoders.psp_encoders import GradualStyleEncoder
import torch

# Class to load and use pSp Encoder for Inversion
class Inverter :
    def __init__(self, num_layers = 18) :
        self.inversion_model = GradualStyleEncoder(num_layers = 50, input_nc = 3, n_styles = num_layers)

    def invert(self,image) :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        latent_code = self.inversion_model(image.unsqueeze(0).to(device=device).float())
        latent_code = latent_code.to(device=device)
        res = latent_code + self.latent_avg.repeat(latent_code.shape[0],1,1)
        return res
    
    def get_keys(self, d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt
    
    def load_model(self, path) :
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path,map_location = device)
        self.latent_avg = ckpt['latent_avg']
        self.latent_avg.to(device=device)
        self.inversion_model.load_state_dict(self.get_keys(ckpt,'encoder'),strict=True)
        self.inversion_model.to(device=device)