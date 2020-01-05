import tensorflow as tf
import numpy as np

_LABELS = ['true', 'false', 'half-true']
# The "distance" for a false-true mistake is 2, and for every other pair - 1
_LABEL_NUMERIC_VALUES = {'false': 0, 'half-true': 1, 'true': 2}

def tf_f1_score(labels, predictions):
    f1s = [0, 0, 0]

    labels = tf.cast(labels, tf.float64)
    predictions = tf.cast(predictions, tf.float64)

    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(predictions * labels, axis=axis)
        TN = tf.count_nonzero((predictions - 1) * (labels - 1), axis=axis)
        FP = tf.count_nonzero(predictions * (labels - 1), axis=axis)
        FN = tf.count_nonzero((predictions - 1) * labels, axis=axis)

        epsilon = tf.cast(tf.keras.backend.epsilon(), 'int64')

        precision = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        f1 = 2 * precision * recall / (precision + recall)

        f1s[i] = tf.reduce_mean(f1)

    weights = tf.reduce_sum(labels, axis=0)
    weights /= tf.reduce_sum(weights)

    f1s[2] = tf.reduce_sum(f1 * weights)

    micro, macro, weighted = f1s
    return {'micro': micro, 'macro': macro, 'weighted': weighted}

def _compute_confusion_matrix(gold_labels, pred_labels):
    """ Computes Confusion Matrix. """
    conf_matrix = {
        'true': {'true': 0, 'false': 0, 'half-true': 0},
        'false': {'true': 0, 'false': 0, 'half-true': 0},
        'half-true': {'true': 0, 'false': 0, 'half-true': 0}
    }

    for pred_label, gold_label in zip(pred_labels, gold_labels):
        pred_label = pred_label.lower()
        true_label = gold_label.lower()
        conf_matrix[true_label][pred_label] += 1

    return conf_matrix


def _compute_accuracy(conf_matrix):
    """ Computes Accuracy. """
    num_claims = sum([sum(row.values()) for row in conf_matrix.values()])
    correct_claims = sum([conf_matrix[l][l] for l in _LABELS])
    if num_claims:
        return correct_claims / num_claims
    else:
        return 0.0


def _compute_macro_f1(conf_matrix):
    """ Computes Macro F1. """
    p = {'true': 0, 'false': 0, 'half-true': 0}
    r = {'true': 0, 'false': 0, 'half-true': 0}

    for label in _LABELS:
        all_predicted = sum([conf_matrix[l][label] for l in _LABELS])

        if all_predicted:
            p[label] = conf_matrix[label][label] / all_predicted
        else:
            p[label] = 0.0
        all_gold = sum([conf_matrix[label][l] for l in _LABELS])

        if all_gold == 0:
            raise ValueError('No instances for class {} found!'.format(label))

        r[label] = conf_matrix[label][label] / all_gold

    f1 = {'true': 0, 'false': 0, 'half-true': 0}
    for label in _LABELS:
        if p[label] + r[label]:
            f1[label] = 2.0 * p[label] * r[label] / (p[label] + r[label])
        else:
            f1[label] = 0.0

    return sum(f1.values()) / len(f1)


def _compute_macro_recall(conf_matrix):
    """ Computes Macro Recall """
    r = {}
    for label in _LABELS:
        all_gold = sum([conf_matrix[label][l] for l in _LABELS])
        if all_gold == 0:
            raise ValueError('No instances for class {} found!'.format(label))

        r[label] = conf_matrix[label][label] / all_gold

    return sum(r.values()) / len(r)


def _compute_mean_absolute_error(conf_matrix):
    """ Computes Mean Absolute Error (MAE). """
    num_claims = sum([sum(row.values()) for row in conf_matrix.values()])
    distance_sum = 0.0
    for gold_label in conf_matrix:
        for pred_label in conf_matrix:
            distance = abs(_LABEL_NUMERIC_VALUES[gold_label] - _LABEL_NUMERIC_VALUES[pred_label])
            distance_sum += conf_matrix[gold_label][pred_label] * distance
    if num_claims:
        return distance_sum / num_claims
    else:
        return 0.0


def _compute_macro_averaged_mae(conf_matrix):
    """ Computes Macro-averaged Mean Absolute Error. """
    mae = {}
    for label in _LABELS:
        all_gold = sum([conf_matrix[label][l] for l in _LABELS])
        distance_sum = 0.0
        gold_label_value = _LABEL_NUMERIC_VALUES[label]
        for pred_label in conf_matrix[label]:
            distance = abs(gold_label_value - _LABEL_NUMERIC_VALUES[pred_label])
            distance_sum += conf_matrix[label][pred_label] * distance
        if all_gold == 0:
            raise ValueError('No instances for class {} found!'.format(label))
        mae[label] = distance_sum / all_gold
    return sum(mae.values()) / len(mae)

def mae(gold_labels, pred_labels):
    return _compute_mean_absolute_error(_compute_confusion_matrix(gold_labels, pred_labels))


def macro_averaged_mae(gold_labels, pred_labels):
    return _compute_macro_averaged_mae(_compute_confusion_matrix(gold_labels, pred_labels))


def accuracy(gold_labels, pred_labels):
    return _compute_accuracy(_compute_confusion_matrix(gold_labels, pred_labels))


def macro_f1(gold_labels, pred_labels):
    return _compute_macro_f1(_compute_confusion_matrix(gold_labels, pred_labels))


def macro_recall(gold_labels, pred_labels):
    return _compute_macro_recall(_compute_confusion_matrix(gold_labels, pred_labels))

metric_callables = {
    'mae': mae,
    'macro_averaged_mae': macro_averaged_mae,
    'accuracy': accuracy,
    'macro_f1': macro_f1,
    'macro_recall': macro_recall,
}
