import numpy as np
from nltk.tag import pos_tag
from nltk.corpus import sentiwordnet as swn, wordnet

from transformers import BaseTransformer

class WordnetSentimentTransformer(BaseTransformer):
    def __init__(self, data):
        self.names = ['pos', 'neg']
        self.features = [self.__compute_score(sent) for sent in data]

    def __compute_score(self, sentence):
        sent_score_pos = []
        sent_score_neg = []
        words = pos_tag(sentence.split())
        for t in words:
            word = t[0]
            tag = t[1]
            new_tag = ''
            if tag.startswith('NN'):
                new_tag = wordnet.NOUN
            elif tag.startswith('J'):
                new_tag = wordnet.ADJ
            elif tag.startswith('V'):
                new_tag = wordnet.VERB
            elif tag.startswith('R'):
                new_tag = wordnet.ADV

            if new_tag != '':
                synsets = list(swn.senti_synsets(word, new_tag))
                score_pos = 0.0
                score_neg = 0.0
                if len(synsets) > 0:
                    for syn in synsets:
                        score_pos += syn.pos_score()
                        score_neg += syn.neg_score()
                    sent_score_pos.append(score_pos / len(synsets))
                    sent_score_neg.append(score_neg / len(synsets))

        pos = sum(sent_score_pos) / len(sent_score_pos) if len(sent_score_pos) > 0 else 0
        neg = sum(sent_score_neg) / len(sent_score_neg) if len(sent_score_neg) > 0 else 0

        return [pos, neg]
