from abc import abstractmethod
import tensorflow as tf
import os
import numpy as np


class DatasetReaderBase:
    def __init__(self, dataset_name, train_size, val_size, total_length, padding_value, data_status):
        self.dataset_name = dataset_name
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = 1. - (train_size + val_size)
        self.data_status = data_status
        self.dataset_path = os.path.join(os.pardir, 'datasets', self.dataset_name.upper(), data_status)
        self.total_length = total_length
        self.padding_value = padding_value

        self.number_of_classes = None  # has to be initialized
        self.number_of_ds_examples = None  # has to be initialized
        self.full_dataset = None  # has to be initialized
        self.train_dataset = None  # has to be initialized
        self.val_dataset = None  # has to be initialized
        self.test_dataset = None  # has to be initialized

    @abstractmethod
    def construct_dataset(self):
        pass

    def _load_all_data_paths(self):
        paths = list()
        for dirpath, _, filenames in os.walk(self.dataset_path):
            paths += [os.path.join(dirpath, filename) for filename in filenames]
        return paths

    def get_number_of_examples(self, set_name='all'):
        if set_name == 'all':
            return self.number_of_ds_examples
        elif set_name == 'train':
            return int(self.train_size * self.number_of_ds_examples)
        elif set_name == 'val':
            return int(self.val_size * self.number_of_ds_examples)
        elif set_name == 'test':
            return int(self.test_size * self.number_of_ds_examples)

    @abstractmethod
    def _load_audio_raw(self, audio, label):
        pass

    @staticmethod
    def _load_audio_wav2vec(file_path, label):
        audio = np.load(bytes.decode(file_path.numpy()))[0]
        return tf.convert_to_tensor(audio, dtype=tf.float64), label

    def get_audio_func(self):
        audio_func = None
        if self.data_status == 'raw_data':
            return self._load_audio_raw
        elif self.data_status == 'wav2vec_data':
            return self._load_audio_wav2vec
        return audio_func

    def get_map_func(self):
        audio_func = self.get_audio_func()
        return lambda *items: tf.py_function(func=audio_func, inp=items, Tout=[tf.float64, tf.int32])

    def train_iterator(self, batch_size=32, shuffle_buffer_size=1024, prefetch=3, num_parallel_calls=-1):
        map_func = self.get_map_func()
        return self.train_dataset \
            .shuffle(buffer_size=shuffle_buffer_size) \
            .map(map_func, num_parallel_calls=num_parallel_calls) \
            .batch(batch_size) \
            .prefetch(prefetch)

    def val_iterator(self, batch_size=32, prefetch=3, num_parallel_calls=-1):
        map_func = self.get_map_func()
        return self.val_dataset \
            .map(map_func, num_parallel_calls=num_parallel_calls) \
            .batch(batch_size) \
            .prefetch(prefetch)

    def test_iterator(self, batch_size=32, prefetch=3, num_parallel_calls=-1):
        map_func = self.get_map_func()
        return self.test_dataset \
            .map(map_func, num_parallel_calls=num_parallel_calls) \
            .batch(batch_size) \
            .prefetch(prefetch)
