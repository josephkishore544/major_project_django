MODEL_PATHS = {
    'text_encoder' : 'models/trained_models/text_encoder.pt',
    'latent_code_decoder' : 'models/trained_models/latent_code_decoder.pt',
    'sbert' : 'models/trained_models/SBERT',
    'inversion' : 'models/trained_models/psp_ffhq_encode.pt',
    'stylegan2' : 'models/trained_models/psp_ffhq_encode.pt',
    'direction_classifier' : 'models/trained_models/direction_classifier.pt',
    'shape_predictor' : 'models/trained_models/shape_predictor_68_face_landmarks.dat'
}

def get_path(key) :
    try :
        path = MODEL_PATHS[key]
    except Exception as e :
        print(e)
    return path