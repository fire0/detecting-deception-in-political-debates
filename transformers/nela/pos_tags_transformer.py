import collections

from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize

from transformers import BaseTransformer

class POSTagsTransformer(BaseTransformer):
    def __init__(self, data, ratio=True):
        pos_tags = [
            'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
            'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'WP$', 'WRB', 'VB', 'VBD', 'VBG',
            'VBN', 'VBP', 'VBZ', 'WDT', 'WP'
        ]

        self.names = pos_tags
        self.features = [self.pos_tags_sentence(text, ratio, pos_tags) for text in data]

    def pos_tags_sentence(self, text, ratio, pos_tags):
        sents = sent_tokenize(text)

        all_tags = []
        for sent in sents:
            tokens = word_tokenize(sent)
            tags = pos_tag(tokens)
            counts = collections.Counter(tags)
            for word, tag in tags:
                all_tags.append(tag)

        counts_norm = [float(all_tags.count(tag)) / len(all_tags) if ratio else all_tags.count(tag) for tag in pos_tags]

        return counts_norm
