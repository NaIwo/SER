from typing import List, Optional, Iterable
import tensorflow as tf
import random

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
            data_labels: DataLabels = DataLabels.from_paths(paths)
            setattr(self, dataset, self._build_datasets_with_x_y(self._get_items_by_indexes(paths, idx),
                                                                 self._get_items_by_indexes(data_labels.labels, idx)))

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
