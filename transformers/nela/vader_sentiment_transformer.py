from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from transformers import BaseTransformer

class VADERSentimentTransformer(BaseTransformer):
    def __init__(self, data, ratio=True):
        analyzer = SentimentIntensityAnalyzer()

        self.names = ['neg', 'neu', 'pos', 'compound']
        self.features = [self.vader_sentiment_sentence(text, analyzer) for text in data]

    def vader_sentiment_sentence(self, text, analyzer):
    	score = analyzer.polarity_scores(text)

        # compound is aggregated
    	return [score['pos'], score['neu'], score['neg']]
