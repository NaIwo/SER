import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Iterable, Optional

from src.datasets.datasets.base_dataset import BaseDataset
from src.datasets.datasets.ravdess.data_details import DataLabels
from src.baselines.config_reader import config


class RavdessDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_name='Ravdess', **kwargs)
        self._construct_datasets()

    def _construct_datasets(self) -> None:
        paths: List = self._load_all_data_paths()
        self.data_labels: DataLabels = DataLabels.from_paths(paths)

        self.number_of_ds_examples = len(self.data_labels.labels)
        self.number_of_classes = len(set(self.data_labels.labels))

        if self.keep_actor_data:
            self.full_dataset = self.build_datasets_with_x_y(paths, self.data_labels.labels)
        else:
            self.full_dataset = self.build_datasets_with_x_y(paths, self.data_labels.labels, self.data_labels.actors)

        if config['ravdess']['split-by-actor']:
            self._construct_datasets_having_actors(paths)
        else:
            self._construct_stratify_train_test_split(paths)

    def _construct_stratify_train_test_split(self, paths: List) -> None:
        indexes = np.array(range(self.number_of_ds_examples))
        if self.get_number_of_examples('train') == 0:  # sklearn can't cope with splitting to empty set
            train_idx, test_idx = [], list(range(self.get_number_of_examples('test')))
        elif self.get_number_of_examples('test') == 0:
            train_idx, test_idx = list(range(self.get_number_of_examples('train'))), []
        else:
            train_idx, test_idx = train_test_split(indexes,
                                                   train_size=self.get_number_of_examples('train'),
                                                   test_size=self.get_number_of_examples('test'),
                                                   random_state=self.seed,
                                                   stratify=self.data_labels.stratify_labels)

        val_idx = set(indexes) - set(train_idx) - set(test_idx)

        self.train_dataset = self.build_datasets_with_x_y(self._get_items_by_indexes(paths, train_idx),
                                                           self._get_items_by_indexes(self.data_labels.labels, train_idx))
        if self.use_augmented_data:
            paths_from_augmentation = [path.replace(self.data_status, self.data_status + '_augmented_vltp')
                                       for path in self._get_items_by_indexes(paths, train_idx)]
            augmentation_set = self.build_datasets_with_x_y(paths_from_augmentation,
                                                             self._get_items_by_indexes(self.data_labels.labels, train_idx))
            self.train_dataset = self.train_dataset.concatenate(augmentation_set)

        self.test_dataset = self.build_datasets_with_x_y(self._get_items_by_indexes(paths, test_idx),
                                                          self._get_items_by_indexes(self.data_labels.labels, test_idx))

        self.val_dataset = self.build_datasets_with_x_y(self._get_items_by_indexes(paths, val_idx),
                                                         self._get_items_by_indexes(self.data_labels.labels, val_idx))

    def _construct_datasets_having_actors(self, paths: List) -> None:
        data_type: str
        for data_type in ['train', 'val', 'test']:
            list_of_actors: List = config['ravdess']['actors'][data_type]
            selected_paths: List = list()
            selected_labels: List = list()
            actors: List = list()
            for label, path in zip(self.data_labels.path_details, paths):
                if label.actor in list_of_actors:
                    selected_labels.append(label.proper_label)
                    selected_paths.append(path)
                    actors.append(label.actor)
            if self.use_augmented_data and data_type == "train":
                augmented_paths = [path.replace(self.data_status, self.data_status + '_augmented_vltp')
                                   for path in selected_paths]
                selected_paths += augmented_paths
                selected_labels += selected_labels
                actors += actors
            if self.keep_actor_data:
                setattr(self, f'{data_type}_dataset', self.build_datasets_with_x_y(selected_paths, selected_labels, actors))
            else:
                setattr(self, f'{data_type}_dataset', self.build_datasets_with_x_y(selected_paths, selected_labels))

    @staticmethod
    def build_datasets_with_x_y(paths: List, labels: List, additional_labels: Optional[List] = None) -> tf.data.Dataset:
        file_paths_dataset = tf.data.Dataset.from_tensor_slices(paths)
        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
        if additional_labels:
            additional_labels_dataset = tf.data.Dataset.from_tensor_slices(additional_labels)
            return tf.data.Dataset.zip((file_paths_dataset, (labels_dataset, additional_labels_dataset)))
        return tf.data.Dataset.zip((file_paths_dataset, labels_dataset))

    @staticmethod
    def _get_items_by_indexes(elements: List, indexes: Iterable) -> List:
        return list(map(elements.__getitem__, indexes))
