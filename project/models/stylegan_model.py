from project.models.stylegan2.model import Generator
import torch

# Class to load and use StyleGAN2
class StyleGAN2 :
    def __init__(self,output_size = 1024, w_dim = 512, mlp = 8) :
        self.generator = Generator(output_size, w_dim, mlp)
        
    def generate(self,latent_codes) :
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        res = self.generator([latent_codes],
                            input_is_latent = True, # input_is_latent = not input_code
                            randomize_noise = True
                            )
        return res[0][0]
    
    def get_keys(self, d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt
    
    def load_model(self, path) :
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path,map_location = device)
        self.generator.load_state_dict(self.get_keys(ckpt,'decoder'),strict=True)
        self.generator.to(device=device)