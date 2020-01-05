from __future__ import absolute_import, division, print_function

import random

import numpy as np
import keras
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from bert_serving.client import BertClient

from utils import flatten, merge_features
from ml_utils import MetricsCallback, plot_keras_model
from estimators.base_estimator import BaseEstimator
from metrics import accuracy, macro_f1, mae, macro_recall
from transformers import WordnetSentimentTransformer, TextstatReadabilityTransformer, LexicalPerSentenceTransformer
from transformers import OpensmileTransformer, AuthorsTransformer, IVectorTransformer
from transformers.nela import LIWCTransformer, VADERSentimentTransformer, POSTagsTransformer

LABELS = ['FALSE', 'HALF-TRUE', 'TRUE']
LABELS_ORDER = {x: y for y, x in enumerate(LABELS)}
RANDOM_SEED = 1234

tf.logging.set_verbosity(tf.logging.INFO)
tf.random.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class MultiInputNN(BaseEstimator):
    def __init__(self):
        self.params = {
            'description': 'Multi Input NN',
            'min_max_scaler_params': {
                'feature_range': (0, 1),
            },
            'tfidf_params': {
                'ngram_range': (1, 4)
            },
            'nn_params': {
                'inputs': [
                    'liwc_authors',
                    'tfidf_ngrams',
                    'ivectors',
                    'bert_cls_token',
                    'opensmile',
                ],
                'epochs': 512,
                'dense_1_neurons': 16,
                'dense_1_dropout':0.5,
                'dense_1_reg': 0.01,
                'dense_2_neurons': 32,
                'dense_2_dropout': 0.5,
                'dense_2_reg': 0.1,
                'early_stopping': {
                    'monitor': 'val_loss',
                    'min_delta': 0,
                    'patience': 5,
                    'mode': 'min',
                }
            },
        }

    def fit(self, debate_data, y, with_grid_search=False):
        self.train_x = self.preprocess(debate_data, is_training=True)
        self.train_y = np.array([LABELS_ORDER[label] for label in flatten(y)])

        self.class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', [0, 1, 2], self.train_y)))

        self.model = self.build_model(self.train_x)

        self.model.compile(
            optimizer=keras.optimizers.SGD(lr=0.005),
            loss=keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy']
        )

        plot_keras_model(self.model, 'model_architecture.png')

    def predict(self, debate_data, proba=False, y=None):
        test_x = self.preprocess(debate_data, is_training=False)
        test_y = np.array([LABELS_ORDER[label] for label in flatten(y)])

        history = self.model.fit(
            x=self.train_x,
            y=self.train_y,
            class_weight=self.class_weights,
            validation_data=(test_x, test_y),
            batch_size=len(self.train_y),
            epochs=self.params['nn_params']['epochs'],
            shuffle=True,
            callbacks=[
                MetricsCallback(num_inputs=len(self.params['nn_params']['inputs']), train_x=self.train_x, train_y=self.train_y, labels=LABELS),
                keras.callbacks.EarlyStopping(
                    monitor=self.params['nn_params']['early_stopping']['monitor'],
                    min_delta=self.params['nn_params']['early_stopping']['min_delta'],
                    patience=self.params['nn_params']['early_stopping']['patience'],
                    mode=self.params['nn_params']['early_stopping']['mode']
                )
            ]
        )

        prediction_probs = list(self.model.predict(test_x))

        predictions_indices = [example_pred_probs.tolist().index(max(example_pred_probs)) for example_pred_probs in prediction_probs]
        predictions = [LABELS[prediction] for prediction in predictions_indices]

        return prediction_probs if proba else predictions

    def build_model(self, train_x):
        input_layers = []
        dropout_layers = []

        for input_type in self.params['nn_params']['inputs']:
            input_layer = keras.layers.Input(shape=train_x[input_type][0].shape, name=input_type)
            input_layers.append(input_layer)

            if input_type in ['tfidf_ngrams']:
                regularizer = keras.regularizers.l1(self.params['nn_params']['dense_1_reg'])
            else:
                regularizer = keras.regularizers.l2(self.params['nn_params']['dense_1_reg'])
            dense_layer = keras.layers.Dense(
                self.params['nn_params']['dense_1_neurons'],
                activation=keras.activations.relu,
                kernel_regularizer=regularizer
            )(input_layer)

            dropout_layer = keras.layers.Dropout(rate=self.params['nn_params']['dense_1_dropout'])(dense_layer)
            dropout_layers.append(dropout_layer)

        inputs = input_layers if len(input_layers) > 1 else input_layers[0]

        concatenated = keras.layers.concatenate(dropout_layers) if len(dropout_layers) > 1 else dropout_layers[0]

        dense_layer = keras.layers.Dense(
            self.params['nn_params']['dense_2_neurons'],
            activation=keras.activations.relu,
            kernel_regularizer=keras.regularizers.l2(self.params['nn_params']['dense_2_reg'])
        )(concatenated)
        dropout_layer = keras.layers.Dropout(rate=self.params['nn_params']['dense_2_dropout'])(dense_layer)

        outputs = keras.layers.Dense(len(LABELS), activation='softmax')(dropout_layer)

        return keras.models.Model(inputs=inputs, outputs=outputs)

    def preprocess(self, debate_data, is_training):
        debates = [x.debate for x in debate_data]
        x_texts = flatten([x.texts for x in debate_data])
        x_authors = flatten([x.authors for x in debate_data])

        if is_training:
            self.opensmile_scaler = MinMaxScaler(**self.params['min_max_scaler_params'])
            self.liwc_scaler = MinMaxScaler(**self.params['min_max_scaler_params'])
            self.tfidf_scaler = MinMaxScaler(**self.params['min_max_scaler_params'])
            self.ivectors_scaler = MinMaxScaler(**self.params['min_max_scaler_params'])
            self.bert_cls_token_scaler = MinMaxScaler(**self.params['min_max_scaler_params'])
            self.wordnet_sentiment_scaler = MinMaxScaler(**self.params['min_max_scaler_params'])
            self.vader_sentiment_scaler = MinMaxScaler(**self.params['min_max_scaler_params'])
            self.lexical_chars_scaler = MinMaxScaler(**self.params['min_max_scaler_params'])
            self.pos_tags_scaler = MinMaxScaler(**self.params['min_max_scaler_params'])
            self.readability_scaler = MinMaxScaler(**self.params['min_max_scaler_params'])
            self.vectorizer = TfidfVectorizer(**self.params['tfidf_params'])
            self.bert_client = BertClient()

        ivectors = IVectorTransformer(debate_data).features

        opensmile = OpensmileTransformer(debates).features
        opensmile = self.opensmile_scaler.fit_transform(opensmile) if is_training else self.opensmile_scaler.transform(opensmile)

        bert_cls_token = self.bert_client.encode(x_texts)

        tfidf_ngrams = self.vectorizer.fit_transform(x_texts).toarray() if is_training else self.vectorizer.transform(x_texts).toarray()
        tfidf_ngrams = self.tfidf_scaler.fit_transform(tfidf_ngrams) if is_training else self.tfidf_scaler.transform(tfidf_ngrams)

        authors = AuthorsTransformer(x_authors).features

        liwc = LIWCTransformer(x_texts).features
        liwc = self.liwc_scaler.fit_transform(liwc) if is_training else self.liwc_scaler.transform(liwc)
        liwc_authors = merge_features(authors, liwc)

        wordnet_sentiment = WordnetSentimentTransformer(x_texts).features
        wordnet_sentiment = self.wordnet_sentiment_scaler.fit_transform(wordnet_sentiment) if is_training else self.wordnet_sentiment_scaler.transform(wordnet_sentiment)

        vader_sentiment = VADERSentimentTransformer(x_texts).features
        vader_sentiment = self.vader_sentiment_scaler.fit_transform(vader_sentiment) if is_training else self.vader_sentiment_scaler.transform(vader_sentiment)

        sentiment = merge_features(wordnet_sentiment, vader_sentiment)

        lexical_chars = LexicalPerSentenceTransformer(x_texts, with_chars=True, with_words=False).features
        lexical_chars = self.lexical_chars_scaler.fit_transform(lexical_chars) if is_training else self.lexical_chars_scaler.transform(lexical_chars)

        pos_tags = POSTagsTransformer(x_texts).features
        pos_tags = self.pos_tags_scaler.fit_transform(pos_tags) if is_training else self.pos_tags_scaler.transform(pos_tags)

        readability = TextstatReadabilityTransformer(x_texts).features
        readability = self.readability_scaler.fit_transform(readability) if is_training else self.readability_scaler.transform(readability)

        lexical = merge_features(liwc, authors, lexical_chars, pos_tags, sentiment, readability)

        return {
            'ivectors': np.array(ivectors),
            'opensmile': np.array(opensmile),
            'bert_cls_token': np.array(bert_cls_token),
            'tfidf_ngrams': np.array(tfidf_ngrams),
            'liwc_authors': np.array(liwc_authors),
            'lexical': np.array(lexical),
            'concatenated': np.array(merge_features(liwc_authors, tfidf_ngrams, ivectors, bert_cls_token, opensmile)),
        }
