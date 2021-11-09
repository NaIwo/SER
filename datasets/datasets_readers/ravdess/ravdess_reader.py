import tensorflow as tf
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

from datasets.datasets_readers.base_reader import DatasetReaderBase
from datasets.datasets_readers.ravdess.data_details import DataLabels


class RavdessReader(DatasetReaderBase):
    def __init__(self, **kwargs):
        super().__init__(dataset_name='Ravdess', **kwargs)

    def _construct_datasets(self):
        paths = self._load_all_data_paths()
        data_labels = DataLabels.from_paths(paths)

        self.number_of_ds_examples = len(data_labels.labels)
        self.number_of_classes = len(set(data_labels.labels))

        self.full_dataset = self._build_datasets_with_x_y(paths, data_labels.labels)

        self._construct_stratify_train_test_split(paths, data_labels)

    def _construct_stratify_train_test_split(self, paths, data_labels):
        indexes = np.array(range(self.number_of_ds_examples))
        train_idx, test_idx = train_test_split(indexes,
                                               train_size=self.get_number_of_examples('train'),
                                               test_size=self.get_number_of_examples('test'),
                                               random_state=self.train_test_seed,
                                               stratify=data_labels.stratify_labels)

        val_idx = set(indexes) - set(train_idx) - set(test_idx)

        self.train_dataset = self._build_datasets_with_x_y(self._get_items_by_indexes(paths, train_idx),
                                                           self._get_items_by_indexes(data_labels.labels, train_idx))

        self.test_dataset = self._build_datasets_with_x_y(self._get_items_by_indexes(paths, test_idx),
                                                          self._get_items_by_indexes(data_labels.labels, test_idx))

        self.val_dataset = self._build_datasets_with_x_y(self._get_items_by_indexes(paths, val_idx),
                                                         self._get_items_by_indexes(data_labels.labels, val_idx))

    @staticmethod
    def _build_datasets_with_x_y(paths, labels):
        file_paths_dataset = tf.data.Dataset.from_tensor_slices(paths)
        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
        return tf.data.Dataset.zip((file_paths_dataset, labels_dataset))

    @staticmethod
    def _get_items_by_indexes(elements, indexes):
        return list(map(elements.__getitem__, indexes))

    def _load_audio_raw(self, file_path, label):
        audio, sample_rate = librosa.load(bytes.decode(file_path.numpy()), sr=self.desired_sampling_rate)
        if self.padding_value is None or self.total_length is None:
            return tf.convert_to_tensor(audio, dtype=tf.float64), label
        else:
            audio_length = len(audio)
            if audio_length > self.total_length:
                audio = audio[:self.total_length]
            elif audio_length < self.total_length:
                audio = np.pad(audio, pad_width=(0, self.total_length - audio_length), mode='constant',
                               constant_values=self.padding_value)
            return tf.convert_to_tensor(audio, dtype=tf.float64), label
