from typing import Tuple, List

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path

from src.config_reader import config

from src.datasets import get_dataset_by_name

class_labels = ["Neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def print_metrics(y_true, y_pred):
    print("Class labels: ", class_labels)
    print("Confusion matrix:", confusion_matrix(y_true, y_pred), sep="\n")
    print("Classification accuracy:", accuracy_score(y_true, y_pred))
    print("Average recall:", recall_score(y_true, y_pred, average="macro"))


def get_correlated_features_indices(data: np.ndarray, threshold: float = 0.9) -> list:
    correlation_matrix = np.corrcoef(data.T)
    coords_of_correlated_features = np.argwhere(np.abs(correlation_matrix) > threshold)
    indices = np.array([x for x in coords_of_correlated_features if x[0] != x[1]])  # removing diagonal
    removed_columns = []
    while indices.shape[0] > 0:
        to_remove = np.argmax(np.bincount(indices.flatten()))
        indices = np.array([x for x in indices if to_remove not in x])
        removed_columns.append(to_remove)
    return removed_columns


def remove_correlated_features(x_train: np.ndarray, x_test: np.ndarray, indices: list) -> Tuple[np.ndarray, np.ndarray]:
    return np.delete(x_train.T, indices, axis=0).T, np.delete(x_test.T, indices, axis=0).T


def plot_results(results: List[float], labels: List[str], metric_name: str,
                 data_type: str, save_plot: bool = False) -> None:
    plt.bar(range(len(results)), results)
    plt.xticks(range(len(results)), labels)
    plt.title(f"{data_type} data")
    plt.ylabel(metric_name)
    if save_plot:
        dir_to_save = config['model']['gemaps-mfcc']['plot-dir']
        if not os.path.exists(dir_to_save):
            Path(dir_to_save).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(dir_to_save, data_type))
    plt.show()


def main():
    dataset_props = config['data']['dataset']
    Dataset = get_dataset_by_name(dataset_props['name'])
    dataset = Dataset(desired_sampling_rate=dataset_props['desired-sampling-rate'],
                      total_length=dataset_props['desired-length'],
                      padding_value=dataset_props['padding-value'],
                      train_size=dataset_props['train-size'],
                      test_size=dataset_props['test-size'],
                      val_size=dataset_props['val-size'],
                      data_status=config['data']['source-name'],
                      train_test_seed=dataset_props['shuffle-seed'],
                      resample_training_set=dataset_props['resample-training-set'])
    x_train, y_train = dataset.get_numpy_dataset(dataset.train_dataset)
    x_test, y_test = dataset.get_numpy_dataset(dataset.test_dataset)
    if len(x_train.shape) == 3:
        x_train = x_train[:, 0]
    if len(x_test.shape) == 3:
        x_test = x_test[:, 0]
    correlated_features_indices = get_correlated_features_indices(x_train)
    x_train, x_test = remove_correlated_features(x_train, x_test, correlated_features_indices)
    standard_scaler = StandardScaler()
    standard_scaler.fit(x_train)
    x_train = standard_scaler.transform(x_train)
    x_test = standard_scaler.transform(x_test)
    results = []

    properties = config['model']['gemaps-mfcc']['classic']
    labels = properties['model-labels']
    for model in (SVC(C=properties['svm']['c'], class_weight='balanced'),
                  RandomForestClassifier(criterion=properties['random-forest']['split-criterion'],
                                         max_depth=properties['random-forest']['max-depth'],
                                         class_weight='balanced'),
                  LogisticRegression(C=properties['logistic-regression']['c'],
                                     max_iter=properties['logistic-regression']['max-iter'],
                                     class_weight='balanced'),
                  MLPClassifier(max_iter=properties['mlp']['max-iter']),
                  DecisionTreeClassifier(class_weight='balanced'),
                  GradientBoostingClassifier(subsample=properties['gbt']['subsample'])):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print_metrics(y_test, y_pred)
        results.append(accuracy_score(y_test, y_pred))
    plot_results(results, labels, "Accuracy", dataset.data_status)


if __name__ == '__main__':
    main()
