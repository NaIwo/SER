from typing import Tuple, List

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
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
    Dataset = get_dataset_by_name(config['data']['dataset']['name'])
    dataset = Dataset(desired_sampling_rate=config['data']['dataset']['desired-sampling-rate'],
                      total_length=config['data']['dataset']['desired-length'],
                      padding_value=config['data']['dataset']['padding-value'],
                      train_size=config['data']['dataset']['train-size'],
                      test_size=config['data']['dataset']['test-size'],
                      val_size=config['data']['dataset']['val-size'],
                      data_status=config['data']['source-name'],
                      train_test_seed=config['data']['dataset']['shuffle-seed'])
    x_train, y_train = dataset.get_numpy_dataset(dataset.train_dataset)
    x_test, y_test = dataset.get_numpy_dataset(dataset.test_dataset)
    correlated_features_indices = get_correlated_features_indices(x_train)
    x_train, x_test = remove_correlated_features(x_train, x_test, correlated_features_indices)
    standard_scaler = StandardScaler()
    standard_scaler.fit(x_train)
    x_train = standard_scaler.transform(x_train)
    x_test = standard_scaler.transform(x_test)
    results = []
    labels = ["SVM", "RF", "LR", "MLP", "DT", "GBT"]
    for model in (SVC(C=100), RandomForestClassifier(criterion='entropy', max_depth=12), LogisticRegression(C=50),
                  MLPClassifier(max_iter=500), DecisionTreeClassifier(), GradientBoostingClassifier(subsample=0.5)):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print_metrics(y_test, y_pred)
        results.append(accuracy_score(y_test, y_pred))
    plot_results(results, labels, "Accuracy", dataset.data_status)


if __name__ == '__main__':
    main()
