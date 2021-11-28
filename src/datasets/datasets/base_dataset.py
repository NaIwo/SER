import random
from abc import abstractmethod
import tensorflow as tf
import os
import numpy as np
import librosa
from pathlib import Path
import warnings
from typing import Optional, List, Callable, Tuple


class BaseDataset:
    def __init__(self,
                 dataset_name: str,
                 desired_sampling_rate: int = 16000,
                 train_size: float = 0.7,
                 test_size: float = 0.3,
                 val_size: Optional[float] = None,
                 total_length: Optional[int] = None,
                 padding_value: Optional[int] = None,
                 data_status: str = 'raw_data',
                 train_test_seed: Optional[int] = None,
                 resample_training_set: bool = False,
                 crop: bool = False,
                 number_of_windows: int = 140):

        self.dataset_name: str = dataset_name

        self.desired_sampling_rate: int = desired_sampling_rate

        self.train_size: float = train_size
        self.test_size: float = test_size
        self.val_size: float = val_size if val_size is not None else 1. - (train_size + test_size)

        self.data_status: str = data_status

        datasets_path: Path = Path(os.path.dirname(os.path.realpath(__file__))).parent
        self.dataset_relative_path: str = os.path.join(datasets_path, 'datasets_files', self.dataset_name.upper(),
                                                       data_status)
        self.total_length: int = total_length
        self.padding_value: float = padding_value

        self.number_of_classes: Optional[int] = None  # has to be initialized
        self.number_of_ds_examples: Optional[int] = None  # has to be initialized
        self.full_dataset: Optional[tf.data.Dataset] = None  # has to be initialized
        self.train_dataset: Optional[tf.data.Dataset] = None  # has to be initialized
        self.val_dataset: Optional[tf.data.Dataset] = None  # has to be initialized
        self.test_dataset: Optional[tf.data.Dataset] = None  # has to be initialized

        self.train_test_seed: int = train_test_seed
        self.resample_training_set = resample_training_set

        self.crop = crop
        self.number_of_windows = number_of_windows

    @abstractmethod
    def _construct_datasets(self) -> None:
        pass

    def _construct_train_test_split(self) -> None:
        self.assert_if_dataset_is_not_none(self.full_dataset)

        full_dataset: tf.data.Dataset = self.full_dataset.shuffle(1000, reshuffle_each_iteration=False,
                                                                  seed=self.train_test_seed)
        self.train_dataset = full_dataset.take(self.get_number_of_examples('train'))
        self.test_dataset = full_dataset.skip(self.get_number_of_examples('train')).take(
            self.get_number_of_examples('test'))
        self.val_dataset = full_dataset.skip(self.get_number_of_examples('train')).skip(
            self.get_number_of_examples('test'))

    def get_number_of_examples(self, set_name: str = 'all') -> int:
        if set_name == 'all':
            return self.number_of_ds_examples
        elif set_name == 'train':
            return int(self.train_size * self.number_of_ds_examples)
        elif set_name == 'val' or set_name == 'dev':
            return int(self.val_size * self.number_of_ds_examples)
        elif set_name == 'test':
            return int(self.test_size * self.number_of_ds_examples)

    def _load_all_data_paths(self, split_name: str = '') -> List:
        paths = list()
        path_to_walk = os.path.join(self.dataset_relative_path, split_name)
        for dirpath, _, filenames in os.walk(path_to_walk):
            paths += [os.path.join(dirpath, filename) for filename in filenames]
        return paths

    def _resample_dataset(self, idx: List, labels: List, dataset_size: int) -> List:
        idx_by_class = []
        resampled_idx = []
        samples_per_class = int(dataset_size / self.number_of_classes)
        for i in range(self.number_of_classes):
            idx_by_class.append([index for index in idx if i == labels[index]])
        for class_idx in idx_by_class:
            resampled_idx += random.choices(class_idx, k=samples_per_class)
        random.shuffle(resampled_idx)
        return resampled_idx

    def train_iterator(self, batch_size: int = 32, shuffle_buffer_size: int = 1024, prefetch: int = 3,
                       num_parallel_calls: int = -1) -> tf.data.Dataset:
        self.assert_if_dataset_is_not_none(self.train_dataset)

        map_func = self.get_map_func()
        return self.train_dataset \
            .shuffle(buffer_size=shuffle_buffer_size) \
            .map(map_func, num_parallel_calls=num_parallel_calls) \
            .batch(batch_size) \
            .prefetch(prefetch)

    def val_iterator(self, batch_size: int = 32, prefetch: int = 3, num_parallel_calls: int = -1) -> tf.data.Dataset:
        self.assert_if_dataset_is_not_none(self.val_dataset)

        map_func = self.get_map_func()
        return self.val_dataset \
            .map(map_func, num_parallel_calls=num_parallel_calls) \
            .batch(batch_size) \
            .prefetch(prefetch)

    def test_iterator(self, batch_size: int = 32, prefetch: int = 3, num_parallel_calls: int = -1) -> tf.data.Dataset:
        self.assert_if_dataset_is_not_none(self.test_dataset)

        map_func = self.get_map_func()
        return self.test_dataset \
            .map(map_func, num_parallel_calls=num_parallel_calls) \
            .batch(batch_size) \
            .prefetch(prefetch)

    def get_map_func(self) -> Callable:
        audio_func = self.get_audio_func()
        return lambda *items: tf.py_function(func=audio_func, inp=items, Tout=[tf.float64, tf.int32])

    def get_audio_func(self) -> Callable:
        if self.data_status == 'raw_data':
            return self._load_audio_raw
        elif self.data_status.startswith('wav2vec_'):
            return self._load_audio_wav2vec
        else:
            return self._load_numpy_features

    def _load_audio_raw(self, file_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        warnings.filterwarnings('ignore')
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

    @staticmethod
    def _load_audio_wav2vec(file_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        audio = np.load(bytes.decode(file_path.numpy()))[0]
        return tf.convert_to_tensor(audio, dtype=tf.float64), label

    def _load_numpy_features(self, file_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        data = np.load(bytes.decode(file_path.numpy()))
        if self.crop:
            data = data[:self.number_of_windows]
        return tf.convert_to_tensor(data, dtype=tf.float64), label

    def get_numpy_dataset(self, dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        self.assert_if_dataset_is_not_none(dataset)

        numpy_dataset = np.array(list(dataset.map(self.get_map_func()).as_numpy_iterator()), dtype=object)
        return np.stack(numpy_dataset[:, 0]), numpy_dataset[:, 1].astype("int32")

    @staticmethod
    def assert_if_dataset_is_not_none(dataset: tf.data.Dataset) -> None:
        assert dataset, 'Dataset has to be initialized first.'
