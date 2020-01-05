import numpy as np
from nltk import pos_tag
from nltk.tokenize import word_tokenize

from utils import print_progress_bar
from transformers import BaseTransformer

# CC | Coordinating conjunction |
# CD | Cardinal number |
# DT | Determiner |
# EX | Existential there |
# FW | Foreign word |
# IN | Preposition or subordinating conjunction |
# JJ | Adjective |
# JJR | Adjective, comparative |
# JJS | Adjective, superlative |
# LS | List item marker |
# MD | Modal |
# NN | Noun, singular or mass |
# NNS | Noun, plural |
# NNP | Proper noun, singular |
# NNPS | Proper noun, plural |
# PDT | Predeterminer |
# POS | Possessive ending |
# PRP | Personal pronoun |
# PRP$ | Possessive pronoun |
# RB | Adverb |
# RBR | Adverb, comparative |
# RBS | Adverb, superlative |
# RP | Particle |
# SYM | Symbol |
# TO | to |
# UH | Interjection |
# VB | Verb, base form |
# VBD | Verb, past tense |
# VBG | Verb, gerund or present participle |
# VBN | Verb, past participle |
# VBP | Verb, non-3rd person singular present |
# VBZ | Verb, 3rd person singular present |
# WDT | Wh-determiner |
# WP | Wh-pronoun |
# WP$ | Possessive wh-pronoun |
# WRB | Wh-adverb |

class LexicalPerSentenceTransformer(BaseTransformer):
    def __init__(self, data, with_words=True, with_chars=False, with_ratio=True):
        transformed = []
        self.names = []
        for index, sent in enumerate(data):
            entry_char = list(sent)
            entry_word = word_tokenize(sent)
            entry_word_tagged = pos_tag(entry_word)

            if with_chars:
                chars, chars_feature_names = self.lexical_chars(entry_char, with_ratio)
                if index == 0:
                    self.names.extend(chars_feature_names)
            else:
                chars = []
            if with_words:
                words, words_feature_names = self.lexical_words(entry_word_tagged, with_ratio)
                if index == 0:
                    self.names.extend(words_feature_names)
            else:
                words = []

            sent_vector = chars + words
            transformed.append(sent_vector)

        self.features = transformed

    def lexical_chars(self, chars, with_ratio):
        char_count = len(chars)

        possible_chars_map = {
            ',': 'comma_count',
            '\n': 'paragraph_count',
            ';': 'semicolon_count',
            ':': 'colon_count',
            ' ': 'spaces_count',
            '\'': 'apostrophes_count',
            '&': 'amp_count'
        }

        possible_chars = possible_chars_map.keys()

        char_analysis = {
            'digits': 0,
            'punctuation_count': 0,
            'comma_count': 0,
            'semicolon_count': 0,
            'colon_count': 0,
            'spaces_count': 0,
            'apostrophes_count': 0,
            'amp_count': 0,
            'parenthesis_count': 0,
            'paragraph_count': 0
        }

        for char in chars:
            if char in possible_chars:
                char_analysis[possible_chars_map[char]] += 1
            elif char.isdigit(): char_analysis['digits'] += 1
            elif char in ['(', ')']: char_analysis['parenthesis_count'] += 1
            if char in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~': char_analysis['punctuation_count'] += 1

        feature_names = list(char_analysis.keys())

        return [char_analysis[key]/char_count if with_ratio else char_analysis[key] for key in feature_names], feature_names

    def lexical_words(self, words_tagged, with_ratio):
        word_count = len(words_tagged)

        word_analysis = {
            'pronouns': 0,
            'prepositions': 0,
            'coordinating_conjunctions': 0,
            'adjectives': 0,
            'adverbs': 0,
            'determiners': 0,
            'interjections': 0,
            'modals': 0,
            'nouns': 0,
            'personal_pronouns': 0,
            'verbs': 0,
            'word_len_gte_six': 0,
            'word_len_two_and_three': 0,
            'avg_word_length': 0,
            'all_caps': 0,
            'capitalized': 0,
            'quotes_count': 0,
        }

        for (word, tag) in words_tagged:
            if tag in ['PRP']: word_analysis['personal_pronouns'] += 1
            if tag.startswith('J'): word_analysis['adjectives'] += 1
            if tag.startswith('N'): word_analysis['nouns'] += 1
            if tag.startswith('V'): word_analysis['verbs'] += 1
            if tag in ['PRP', 'PRP$', 'WP', 'WP$']: word_analysis['pronouns'] += 1
            elif tag in ['IN']: word_analysis['prepositions'] += 1
            elif tag in ['CC']: word_analysis['coordinating_conjunctions'] += 1
            elif tag in ['RB', 'RBR', 'RBS']: word_analysis['adverbs'] += 1
            elif tag in ['DT', 'PDT', 'WDT']: word_analysis['determiners'] += 1
            elif tag in ['UH']: word_analysis['interjections'] += 1
            elif tag in ['MD']: word_analysis['modals'] += 1
            if len(word) >= 6: word_analysis['word_len_gte_six'] += 1
            elif len(word) in [2, 3]: word_analysis['word_len_two_and_three'] += 1
            word_analysis['avg_word_length'] += len(word)
            if word.isupper(): word_analysis['all_caps'] += 1
            elif word[0].isupper(): word_analysis['capitalized'] += 1
            word_analysis['quotes_count'] += word.count('"') + word.count('`') + word.count('\'')

        feature_names = list(word_analysis.keys())

        return [word_analysis[key]/word_count if with_ratio else word_analysis[key] for key in feature_names], feature_names
