from utils import load_ivector_features, flatten
from transformers import BaseTransformer

class IVectorTransformer(BaseTransformer):
    def __init__(self, debates):
        audio_features, attributes = load_ivector_features(debates)
        self.names = attributes
        self.features = flatten(audio_features)
