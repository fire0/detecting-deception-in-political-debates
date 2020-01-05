from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from bert_serving.client import BertClient

from estimators.base_estimator import BaseEstimator
from utils import get_n_jobs, merge_features, flatten, grid_search_log_reg_c
from transformers import OpensmileTransformer, AuthorsTransformer, IVectorTransformer
from transformers.nela import LIWCTransformer

class LogRegConcatenated(BaseEstimator):
    def __init__(self):
        self.params = {
            'description': 'Logistic Regression with LIWC + Opensmile + TFIDF n-gram + IVectors + Authors + BERT_CLS_TOKEN concatenated features',
            'min_max_scaler_params': {
                'feature_range': (0, 1),
            },
            'tfidf_params': {
                'ngram_range': (1, 4)
            },
            'log_reg': {
                'penalty': 'l2',
                'dual': False,
                'tol': 0.0001,
                'C': 0.0001,
                'fit_intercept': True,
                'intercept_scaling': 1,
                'class_weight': 'balanced',
                'solver': 'sag',
                'max_iter': 20000,
                'multi_class': 'multinomial',
                'random_state': 0,
                'verbose': 0,
                'warm_start': False,
                'n_jobs': get_n_jobs()
            }
        }

        self.bert_client = BertClient()

    def fit(self, debate_data, y, with_grid_search=True):
        debates = [x.debate for x in debate_data]
        x_texts = flatten([x.texts for x in debate_data])
        x_authors = flatten([x.authors for x in debate_data])
        y = flatten(y)

        self.ivectors_scaler = MinMaxScaler(**self.params['min_max_scaler_params'])
        self.opensmile_scaler = MinMaxScaler(**self.params['min_max_scaler_params'])
        self.bert_cls_token_scaler = MinMaxScaler(**self.params['min_max_scaler_params'])
        self.tfidf_scaler = MinMaxScaler(**self.params['min_max_scaler_params'])
        self.liwc_scaler = MinMaxScaler(**self.params['min_max_scaler_params'])
        self.authors_scaler = MinMaxScaler(**self.params['min_max_scaler_params'])
        self.vectorizer = TfidfVectorizer(**self.params['tfidf_params'])

        ivectors = IVectorTransformer(debate_data).features
        ivectors = self.ivectors_scaler.fit_transform(ivectors)

        opensmile = OpensmileTransformer(debates).features
        opensmile = self.opensmile_scaler.fit_transform(opensmile)

        bert_cls_token = self.bert_client.encode(x_texts)
        bert_cls_token = self.bert_cls_token_scaler.fit_transform(bert_cls_token)

        tfidf_ngrams = self.vectorizer.fit_transform(x_texts).toarray()
        tfidf_ngrams = self.tfidf_scaler.fit_transform(tfidf_ngrams)

        liwc = LIWCTransformer(x_texts).features
        liwc = self.liwc_scaler.fit_transform(liwc)

        authors = AuthorsTransformer(x_authors).features
        authors = self.authors_scaler.fit_transform(authors)

        train_x = merge_features(ivectors, opensmile, bert_cls_token, tfidf_ngrams, liwc, authors)

        pipe = Pipeline([
            ('log_reg', LogisticRegression(**self.params['log_reg']))
        ])

        if with_grid_search:
            self.model = grid_search_log_reg_c(pipe, train_x, y, debate_data, 'macro_recall')
        else:
            self.model = pipe
            self.model.fit(train_x, y)

    def predict(self, debate_data, proba=False):
        debates = [x.debate for x in debate_data]
        x_texts = flatten([x.texts for x in debate_data])
        x_authors = flatten([x.authors for x in debate_data])

        ivectors = IVectorTransformer(debate_data).features
        ivectors = self.ivectors_scaler.transform(ivectors)

        opensmile = OpensmileTransformer(debates).features
        opensmile = self.opensmile_scaler.transform(opensmile)

        bert_cls_token = self.bert_client.encode(x_texts)
        bert_cls_token = self.bert_cls_token_scaler.transform(bert_cls_token)

        tfidf_ngrams = self.vectorizer.transform(x_texts).toarray()
        tfidf_ngrams = self.tfidf_scaler.transform(tfidf_ngrams)

        liwc = LIWCTransformer(x_texts).features
        liwc = self.liwc_scaler.transform(liwc)

        authors = AuthorsTransformer(x_authors).features
        authors = self.authors_scaler.transform(authors)

        test_x = merge_features(ivectors, opensmile, bert_cls_token, tfidf_ngrams, liwc, authors)

        return self.model.predict_proba(test_x) if proba else self.model.predict(test_x)
