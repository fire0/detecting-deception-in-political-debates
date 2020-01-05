import pprint
import logging
import os
import json
import time
import uuid
import errno
import arff
import csv
import itertools
import re
from argparse import ArgumentParser
from itertools import zip_longest, chain
from collections import OrderedDict, namedtuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, train_test_split
from sklearn.metrics import confusion_matrix, make_scorer

from metrics import mae, macro_averaged_mae, accuracy, macro_f1, macro_recall

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Reds):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Confusion matrix")
    else:
        print('Confusion matrix')

    np.set_printoptions(precision=2)

    print(pd.DataFrame(cm, index=['true:' + label for label in classes], columns=['pred:' + label for label in classes]).round(2))

    fig, ax = plt.subplots()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    ax.xaxis.tick_top()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=30, horizontalalignment="center", verticalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax.xaxis.set_label_position('top')

    plt.tight_layout()
    plt.show()

# 6373 features
def load_audio_features(debates):
    debate_audio_data = []
    debate_audio_attributes = None
    for debate in debates:
        audio_features_dir = 'data/debates/%s/audio_features' % debate
        debate_data = []
        utterance_id_key = lambda filename: int(re.search('audio_%s_(.*).arff' % debate, filename).group(1))
        for audio_segment in sorted(os.listdir(audio_features_dir), key=utterance_id_key):
            arff_file_data = arff.load(open(os.path.join(audio_features_dir, audio_segment)))
            # remove name and class attributes
            attributes = [attribute for attribute, _ in arff_file_data['attributes']][1:-1]
            statement_data = arff_file_data['data'][0][1:-1]
            debate_data.append(statement_data)
            debate_audio_attributes = attributes
        debate_audio_data.append(debate_data)

    return debate_audio_data, debate_audio_attributes


def load_ivector_features(debates):
    ivectors = np.load('data/ivectors/debates_mfcc_fft400_hop160_vad_cmn.1.eb.npy')

    with open('data/ivectors/wav.scp') as input_file:
        orders = [line.rstrip('\n').split()[1] for line in input_file.readlines()]
        orders = [os.path.split(line) for line in orders]
        orders = [filename[len('audio_'):-len('.wav')] for path, filename in orders if filename.endswith('.wav') and filename.startswith('audio_')]
        orders = [filename.rsplit('_', 1) for filename in orders]

    debate_audio_data = []
    for debate in debates:
        ivectors_indices = [orders.index([debate.debate, claim_id]) for claim_id in debate.ids]
        debate_ivectors = np.float64(ivectors[ivectors_indices])
        debate_audio_data.append(debate_ivectors)

    debate_ivectors_attributes = []
    return debate_audio_data, debate_ivectors_attributes


def get_eval_data(debates, audio=True, text=True, merge=False):
    debates_data, labels = [], []
    DebateData = namedtuple('DebateData', 'debate texts authors ids')

    Statement = namedtuple('Statement', 'utterance_id speaker_id text claim_number normalized_claim label')
    for debate in debates:
        transcription_path = 'data/debates/%s/transcription.txt' % debate
        with open(transcription_path) as transcription:
            content = [Statement(*x.rstrip('\n').split('	')) for x in transcription.readlines()]

        filtered_data = list(filter(lambda statement: statement.label in ['TRUE', 'HALF-TRUE', 'FALSE'], content))

        texts = [process_utterance(row.text) for row in filtered_data]
        speaker_ids = [row.speaker_id for row in filtered_data]
        utterance_ids = [row.utterance_id for row in filtered_data]
        debates_data.append(DebateData(debate, texts, speaker_ids, utterance_ids))

        labels.append([row.label for row in filtered_data])

    return debates_data, labels

def process_utterance(text):
    return ' '.join([seg.group().lower().replace("’", "'") for seg in re.finditer(r'(\w|\’\w|\'\w)+', text, re.UNICODE)])

def merge_transformers(*transformers):
    features_lists = [transformer.features for transformer in transformers]
    names_lists = [transformer.names for transformer in transformers]

    features = [list(chain.from_iterable(x)) for x in zip(*list(features_lists))]
    names = flatten(names_lists)
    return features, names


def merge_features(*lists):
    return [list(chain.from_iterable(x)) for x in zip(*list(lists))]


# If shuffle=False then stratify must be None.
def split_data(x, y, ratio=0.80, shuffle=True, stratify_by_labels_ratio=True, random_state=42):
    train_test_split_args = {
        'shuffle': shuffle,
        'stratify': y if stratify_by_labels_ratio and shuffle else None,
        'random_state': random_state,
    }

    return train_test_split(x, y, **train_test_split_args)


def aggregate_results(clf_params, cv=None, val=None, test=None):
    results = {
        'estimator': clf_params,
        'cross_validation': cv,
        'validation': val,
        'test': test,
    }

    results_json = json.dumps(OrderedDict(**results), indent=4)
    print(results_json)

    return results_json


def pretty_print_results(y, y_predicted, precise=False):
    text_colors = ['1;31', '1;32', '1;33']
    def coloring_precise(x, y):
        if x == y:
            code = text_colors[1]
        else:
            code = text_colors[0]
        return code

    def coloring_relative(x, y):
        if abs(x - y) == 0:
            code = text_colors[1]
        elif abs(x - y) == 1:
            code = text_colors[2]
        elif abs(x - y) == 2:
            code = text_colors[0]
        return code

    whole_print = []
    for x, y in zip(y, y_predicted):
        code = coloring_precise(x, y) if precise else coloring_relative(x, y)
        whole_print.append('\x1b[%sm%s\x1b[0m' % (code, y))
    return '-'.join(whole_print)


def write_results_to_file(results):
    results_path = config_local().get('results_file', None)

    if not results_path:
        print('No results_file specified in the local configuration!')
        return

    results_dir = os.path.dirname(results_path)
    if results_dir:
        create_directory_if_not_exist(results_dir)
    file_prepend(results_path, results, '\n' + '='*100 + '\n')


def create_directory_if_not_exist(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def file_prepend(file_path, content, separator=''):
    try:
        with open(file_path, 'r+') as output:
            old_content = output.read()
            if not old_content:
                separator = ''
            new_content = content + separator + old_content
            output.seek(0, 0)
            output.write(new_content)
    except FileNotFoundError:
        with open(file_path, 'w') as output:
            output.write(content)


def config_local():
    with open('config.json', 'r') as config_json:
        return json.load(config_json)


def round_np_scores(np_array, p=None):
    return [round(x, p) for x in np_array.tolist()]


def humanize_time(secs):
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)

    return '%02d:%02d:%02d' % (hours, mins, secs)


def print_progress_bar(iteration, total, description='', decimals=1, bar_length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(bar_length * iteration // total)
    bar = fill * filled_length + '-' * (bar_length - filled_length)

    print('\r |%s| %s%% | %s' % (bar, percent, description), end='\r')

    if iteration == total:
        print()


def chunker(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n

    return list(zip_longest(*args, fillvalue=fillvalue))


def get_n_jobs():
    with open('config.json', 'r') as rc:
        return json.load(rc).get('n_jobs', 1)


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("-i", dest="input_dir", help="input_dir", metavar="FILE")
    parser.add_argument("-o", dest="output_dir", help="output_dir", metavar="FILE")
    args = parser.parse_args()

    return args.input_dir, args.output_dir


def grid_search_log_reg_c(estimator, train_x, y, debate_group_data = None, refit_metric = 'macro_recall'):
    groups = [[index+1]*len(group) for index, group in enumerate([x.authors for x in debate_group_data])] if debate_group_data else None

    param_grid = [
        {
            'log_reg__C': [0.0001, 0.001, 0.01, 0.1, 0.3, 0.6, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        },
    ]

    scoring = {
        'mae': make_scorer(mae, greater_is_better=False),
        'macro_averaged_mae': make_scorer(macro_averaged_mae, greater_is_better=False),
        'accuracy': make_scorer(accuracy, greater_is_better=True),
        'macro_f1': make_scorer(macro_f1, greater_is_better=True),
        'macro_recall': make_scorer(macro_recall, greater_is_better=True),
    }

    cv_op = LeaveOneGroupOut() if groups else 3

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_op,
        refit=scoring[refit_metric],
        iid=False,
        return_train_score=True,
        error_score='raise',
        verbose=0,
        n_jobs=get_n_jobs()
    )
    search.fit(train_x, y, groups) if groups else search.fit(train_x, y)
    pprint.pprint(search.best_params_)
    pprint.pprint(search.best_score_)

    return search.best_estimator_

def update_dict(params, keys, value):
    if len(keys) == 1:
        params[keys[0]] = value
    else:
        update_dict(params[keys[0]], keys[1:], value)


def flatten(list):
    return [item for sublist in list for item in sublist]


def merge_dataset_nested_lists(a, b):
    return [list(chain.from_iterable(x)) for x in zip(a, b)]
