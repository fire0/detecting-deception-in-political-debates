from utils import load_audio_features, flatten
from transformers import BaseTransformer

class OpensmileTransformer(BaseTransformer):
    def __init__(self, debates):
        audio_features, attributes = load_audio_features(debates)
        self.names = attributes
        self.features = flatten(audio_features)
