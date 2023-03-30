import torch
import numpy as np

latent_direction_file_name = {
    'age' : 'age.npy',
    'gender' : 'gender.npy',
    'smile' : 'smile.npy',
    'eye_open' : 'eyes_open.npy',
    'mouth_open' : 'mouth_open.npy',
    'pitch' : 'pitch.npy',
    'nose' : 'nose_ratio.npy',
}

latent_direction_folder_path = 'models/trained_models/latentdirections/'

# Class to perform Latent Manipulation
# Add a text classifier
class LatentManipulator :
    def __init__(self) :
        pass
    
    def load_latent_direction(self, attribute) :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        file_name = latent_direction_file_name[attribute]
        latent_direction_file_path = latent_direction_folder_path + file_name
        latent_direction = torch.from_numpy(np.load(latent_direction_file_path))
        latent_direction = latent_direction.unsqueeze(0).to(device = device).float()
        return latent_direction

    def manipulate_latent(self, latent_code, attribute, effect, strength = 8) :
        latent_direction = self.load_latent_direction(attribute)
        if effect == 'increase' :
            multiplier = 1
        elif effect == 'decrease' :
            multiplier = -1
        alpha = multiplier*strength
        w_plus = latent_code + alpha*latent_direction
        return w_plus