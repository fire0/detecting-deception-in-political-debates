from transformers import BaseTransformer

class AuthorsTransformer(BaseTransformer):
    def __init__(self, data):
        self.names = ['AUTHOR_TRUMP', 'AUTHOR_CLINTON', 'AUTHOR_OTHER']

        transformed = []
        for author in data:
            is_trump = 1 if 'TRUMP' in author else 0
            is_clinton = 1 if 'CLINTON' in author else 0
            is_other = 1 if is_trump == 0 and is_clinton == 0 else 0
            transformed.append([is_trump, is_clinton, is_other])

        self.features = transformed
