from typing import List, Optional, Iterable
import tensorflow as tf
import random
import glob

from src.baselines.config_reader import config
from .data_details import DataLabels
from src.datasets.datasets.base_dataset import BaseDataset


class MeldDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__("Meld", **kwargs)
        self.number_of_classes: int = 7

        self.number_of_train_examples: Optional[int] = None
        self.number_of_test_examples: Optional[int] = None
        self.number_of_dev_examples: Optional[int] = None

        self._construct_datasets()

    def _construct_datasets(self) -> None:
        names = ['train', 'test', 'dev']
        numbers = ['number_of_train_examples', 'number_of_test_examples', 'number_of_dev_examples']
        datasets = ['train_dataset', 'test_dataset', 'val_dataset']

        for name, var, dataset in zip(names, numbers, datasets):
            paths: List = self._load_all_data_paths(split_name=name)
            setattr(self, var, len(paths))
            idx = random.sample(range(getattr(self, var)), self.get_number_of_examples(name))
            data_labels: DataLabels = DataLabels.from_paths(paths, source=name)
            if dataset == 'train_dataset' and self.resample_training_set:
                idx = self._resample_dataset(idx, data_labels.labels, getattr(self, var))
            idx_before = idx
            idx = self._filter_unsupported_indexes(idx, data_labels)
            setattr(self, var, getattr(self, var) - len(set(idx_before) - set(idx)))
            dataset_paths, dataset_labels = self._get_items_by_indexes(paths, idx), \
                                             self._get_items_by_indexes(data_labels.labels, idx)
            setattr(self, dataset, self._build_datasets_with_x_y(dataset_paths, dataset_labels))
            if self.use_augmented_data and name == 'train':
                augmentation_set = self._get_dataset_with_augmented_data(dataset_paths, dataset_labels)
                setattr(self, dataset, getattr(self, dataset).concatenate(augmentation_set))

    def _get_dataset_with_augmented_data(self, paths: list, labels: list) -> tf.data.Dataset:
        augmented_dir = self.dataset_relative_path.replace(self.data_status, self.data_status + '_augmented_vltp')
        aug_paths = glob.glob(f"{augmented_dir}/**/*.*", recursive=True)
        paths_from_augmentation = []
        labels_from_augmentation = []
        for path, label in zip(paths, labels):
            corresponding_paths = self._get_corresponding_paths(path, aug_paths)
            paths_from_augmentation += corresponding_paths
            labels_from_augmentation += [label] * len(corresponding_paths)
        return self._build_datasets_with_x_y(paths_from_augmentation, labels_from_augmentation)

    def _filter_unsupported_indexes(self, indexes: List, data_labels: DataLabels) -> List:
        return [idx for idx in indexes if data_labels.path_details[idx].supported]

    def _get_corresponding_paths(self, path, aug_paths):
        path_to_augmented_data = path.replace(self.data_status, self.data_status + '_augmented_vltp')
        path_without_file_extension = path_to_augmented_data[:path_to_augmented_data.rfind('.')] + '_'
        return [aug_path for aug_path in aug_paths if path_without_file_extension in aug_path]

    def get_number_of_examples(self, set_name: str = 'all') -> int:
        if set_name == 'all':
            return self.number_of_train_examples + self.number_of_test_examples + self.number_of_dev_examples
        elif set_name == 'train':
            return int(self.train_size * self.number_of_train_examples)
        elif set_name == 'val' or set_name == 'dev':
            return int(self.val_size * self.number_of_dev_examples)
        elif set_name == 'test':
            return int(self.test_size * self.number_of_test_examples)

    @staticmethod
    def _build_datasets_with_x_y(paths: List, labels: List) -> tf.data.Dataset:
        file_paths_dataset = tf.data.Dataset.from_tensor_slices(paths)
        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
        return tf.data.Dataset.zip((file_paths_dataset, labels_dataset))

    @staticmethod
    def _get_items_by_indexes(elements: List, indexes: Iterable) -> List:
        return list(map(elements.__getitem__, indexes))
