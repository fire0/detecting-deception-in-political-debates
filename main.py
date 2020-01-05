import time
import statistics

import numpy as np
from sklearn.metrics import confusion_matrix

from estimators import MultiInputNN
from metrics import metric_callables
from utils import get_eval_data, flatten, aggregate_results, write_results_to_file, humanize_time, config_local
from utils import pretty_print_results, plot_confusion_matrix

# 94 train, 192 test
def main(
    estimators=[MultiInputNN],
    train_debates=[
        '1st_presidential',
        '2nd_presidential',
        'vice_presidential',
    ],
    test_debates=[
        '3rd_presidential',
        '9th_democratic',
        'trump_acceptance',
        'trump_at_wef',
        'trump_address_to_congress',
        'trump_at_tax_reform_event',
        'trump_miami_speech',
    ],
    eval_cross_validation=False,
    eval_validation=True,
    metrics=['accuracy', 'mae', 'macro_averaged_mae', 'macro_f1', 'macro_recall']
):
    train_debate_data, train_labels = get_eval_data(train_debates)
    test_debate_data, test_labels = get_eval_data(test_debates)

    eval_metrics = {k: metric_callables[k] for k in metrics}

    for estimator in estimators:
        clf = estimator() if estimator else RandomBaseline()

        print('Running Estimator:', clf.__class__.__name__, clf.params['description'])
        cv_result = cross_validation_loo(estimator, train_debate_data + test_debate_data, train_labels + test_labels, eval_metrics) if eval_cross_validation else None
        val_result = evaluate_model(estimator, train_debate_data, train_labels, test_debate_data, test_labels, eval_metrics) if eval_validation else None
        results = aggregate_results(clf_params=clf.params, cv=cv_result, val=val_result)

        if config_local().get('persist_results', False): write_results_to_file(results)

def cross_validation_loo(estimator, debate_data, y, metrics):
    print('Cross-validating model... Samples size:', len(y))
    t_start = time.time()
    debates_len = len(debate_data)
    fold_results = []
    for index, (test_debate_data, test_labels) in enumerate(zip(debate_data, y)):
        train_debate_data = debate_data[0:index] + debate_data[index + 1:debates_len + 1]
        train_labels = y[0:index] + y[index + 1:debates_len + 1]

        result = evaluate_model(estimator, train_debate_data, train_labels, [test_debate_data], [test_labels], metrics)
        print(result)
        fold_results.append(result)

    cv_results = {}
    for key in fold_results[0].keys():
        cv_results[key] = {}
        values = [x[key] for x in fold_results]
        cv_results[key]['mean'] = round(statistics.mean(values), 4)
        cv_results[key]['std'] = round(statistics.stdev(values), 4)

    for key, value in cv_results.items():
        print('%s: %s (mean), %s (std)' % (key, str(value['mean']), str(value['std'])))

    t_end = time.time()
    print('Cross-validating finished... Time taken:', humanize_time(t_end - t_start))

    return cv_results

def evaluate_model(estimator, train_debate_data, train_y, test_debate_data, test_y, metrics):
    clf = estimator() if estimator else RandomBaseline()
    train_y_flat = flatten(train_y)
    test_y_flat = flatten(test_y)

    print('Evaluating model... Train/Test samples: %d / %d' % (len(train_y_flat), len(test_y_flat)))

    t_start = time.time()
    clf.fit(train_debate_data, train_y)
    predictions = clf.predict(test_debate_data, y=test_y)

    t_end = time.time()

    print('Evaluating finished... Time taken:', humanize_time(t_end - t_start))

    if type(predictions) is np.ndarray: predictions = predictions.tolist()

    results = {k: round(metric_callable(test_y_flat, predictions), 4) for k, metric_callable in metric_callables.items()}

    pretty_print_predictions(test_debate_data, predictions, test_y_flat)
    confusion_matrix(predictions, test_y_flat)

    return {
        **results,
        'time': t_end - t_start,
        'train_size': len(train_y_flat),
        'validation_size': len(test_y_flat),
    }

def confusion_matrix(predictions, y):
    class_names = ['FALSE', 'HALF-TRUE', 'TRUE']
    conf_matrix = confusion_matrix(y, predictions, labels=class_names)
    plot_confusion_matrix(conf_matrix, classes=class_names, normalize=False)

def pretty_print_predictions(test_debate_data, predictions, test_y):
    test_debates_size_mask = [len(debate_data.ids) for debate_data in test_debate_data]
    test_debates = [x.debate for x in test_debate_data]
    class_ordering = {'FALSE': 0, 'HALF-TRUE': 1, 'TRUE': 2}
    test_y_order = [class_ordering[label] for label in test_y]
    predictions_order = [class_ordering[label] for label in predictions]
    counter = 0
    for debate_index, i in enumerate(test_debates_size_mask):
        print(test_debates[debate_index], pretty_print_results(test_y_order[counter:i+counter], predictions_order[counter:i+counter], precise=True))
        counter += i

if __name__ == '__main__':
    t_start = time.time()
    main()
    t_end = time.time()
    print("Time taken: %s" % humanize_time(t_end - t_start))
