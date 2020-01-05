from textstat.textstat import textstat

from utils import print_progress_bar
from transformers import BaseTransformer

class TextstatReadabilityTransformer(BaseTransformer):
    def __init__(self, data, indices = ['flesch_reading_ease', 'smog_index', 'flesch_kincaid_grade', 'coleman_liau_index', \
        'automated_readability_index', 'dale_chall_readability_score', 'difficult_words', \
        'linsear_write_formula', 'gunning_fog']):
        self.names = indices

        transformed = []

        data_length = len(data)

        for i, doc in enumerate(data):
            features = [getattr(textstat, index)(doc) for index in indices]
            transformed.append(features)

            print_progress_bar(i + 1, data_length, description = 'readability')

        self.features = transformed
