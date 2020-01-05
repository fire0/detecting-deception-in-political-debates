import os
import string

from nltk import word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from transformers import BaseTransformer

class LIWCTransformer(BaseTransformer):
    def __init__(self, data):
        cat_dict, stem_dict, counts_dict = self.load_LIWC_dictionaries()
        liwc_cats = [cat_dict[cat] for cat in cat_dict]

        self.names = liwc_cats
        self.features = [self.liwc_sentence(text, cat_dict, stem_dict, counts_dict) for text in data]

    def liwc_sentence(self, text, cat_dict, stem_dict, counts_dict):
        quotes, Exclaim, AllPunc, allcaps = self.stuff_LIWC_leftout(text)
        counts_norm_liwc, liwc_cats = self.LIWC_features(text, cat_dict, stem_dict, counts_dict)

        return counts_norm_liwc

    def stuff_LIWC_leftout(self, text):
    	puncs = set(string.punctuation)
    	tokens = word_tokenize(text)
    	quotes = tokens.count("\"") + tokens.count('``') + tokens.count("''")
    	Exclaim = tokens.count("!")
    	AllPunc = 0
    	for p in puncs:
    		AllPunc += tokens.count(p)
    	words_upper = 0
    	for w in tokens:
    		if w.isupper():
    			words_upper += 1
    	try:
    		allcaps = float(words_upper) / len(tokens)
    	except err:
    		print(err)
    	return (float(quotes) / len(tokens)) * 100, (float(Exclaim) / len(tokens)) * 100, (float(AllPunc) / len(tokens)) * 100, allcaps

    def load_LIWC_dictionaries(self, filepath="./data/external/"):
    	cat_dict = {}
    	stem_dict = {}
    	counts_dict = {}
    	with open(os.path.join(filepath, "LIWC2007.dic")) as raw:
    		raw.readline()
    		for line in raw:
    			if line.strip() == "%":
    				break
    			line = line.strip().split()
    			cat_dict[line[0]] = line[1]
    			counts_dict[line[0]] = 0
    		for line in raw:
    			line = line.strip().split()
    			stem_dict[line[0]] = [l.replace("*", "") for l in line[1:]]
    	return cat_dict, stem_dict, counts_dict

    def LIWC_features(self, text, cat_dict, stem_dict, counts_dict):
        for key in counts_dict:
            counts_dict[key] = 0
        tokens = word_tokenize(text)
        word_count = len(tokens)

        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        stemed_tokens = []
        for word, tag in pos_tag(tokens):
            try:
                word_stem = lemmatizer.lemmatize(word, tag)
            except:
                word_stem = stemmer.stem(word)
            stemed_tokens.append(word_stem)

        for stem in stem_dict:
            count = stemed_tokens.count(stem.replace("*", ""))
            if count == 0: continue
            for cat in stem_dict[stem]:
                counts_dict[cat] += count
        counts_norm = [float(counts_dict[cat]) / word_count * 100 for cat in counts_dict]
        cats = [cat_dict[cat] for cat in cat_dict]
        return counts_norm, cats
