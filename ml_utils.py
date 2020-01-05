import keras
import numpy as np

from utils import flatten
from metrics import macro_f1, macro_recall, mae, macro_averaged_mae

class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, num_inputs, train_x, train_y, labels):
        super(keras.callbacks.Callback, self).__init__()
        self.num_inputs = num_inputs
        self.train_x = train_x
        self.train_y = train_y
        self.labels = labels

    def on_epoch_end(self, epoch, logs={}):
        train_predict = np.asarray(self.model.predict(self.train_x))

        train_predictions_indices = [example_pred_probs.tolist().index(max(example_pred_probs)) for example_pred_probs in train_predict]
        train_predictions = [self.labels[prediction] for prediction in train_predictions_indices]

        train_targets = [self.labels[target] for target in self.train_y]

        logs['macro_f1'] = macro_f1(train_targets, train_predictions)
        logs['macro_recall'] = macro_recall(train_targets, train_predictions)
        logs['mae'] = mae(train_targets, train_predictions)
        logs['macro_averaged_mae'] = macro_averaged_mae(train_targets, train_predictions)

        val_data = self.validation_data[:self.num_inputs]

        val_predict = np.asarray(self.model.predict(val_data))

        predictions_indices = [example_pred_probs.tolist().index(max(example_pred_probs)) for example_pred_probs in val_predict]
        predictions = [self.labels[prediction] for prediction in predictions_indices]

        val_targ = flatten(self.validation_data[self.num_inputs])
        targets = [self.labels[target] for target in val_targ]

        logs['val_macro_f1'] = macro_f1(targets, predictions)
        logs['val_macro_recall'] = macro_recall(targets, predictions)
        logs['val_mae'] = mae(targets, predictions)
        logs['val_macro_averaged_mae'] = macro_averaged_mae(targets, predictions)

def plot_keras_model(model, filename=None):
    keras.utils.plot_model(model, to_file=filename, show_shapes=True)
