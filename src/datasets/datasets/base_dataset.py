import random
from abc import abstractmethod
import tensorflow as tf
import os
import numpy as np
import librosa
from pathlib import Path
import warnings
from typing import Optional, List, Callable, Tuple, Union


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
                 seed: Optional[int] = None,
                 resample_training_set: bool = False,
                 crop: bool = False,
                 number_of_windows: int = 140,
                 use_augmented_data: bool = False,
                 keep_actor_data: bool = False):

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

        self.seed: int = seed
        self.resample_training_set = resample_training_set

        self.crop = crop
        self.number_of_windows = number_of_windows

        self.use_augmented_data = use_augmented_data
        self.keep_actor_data = keep_actor_data

    @abstractmethod
    def _construct_datasets(self) -> None:
        pass

    def _construct_train_test_split(self) -> None:
        self.assert_if_dataset_is_not_none(self.full_dataset)

        full_dataset: tf.data.Dataset = self.full_dataset.shuffle(1000, reshuffle_each_iteration=False,
                                                                  seed=self.seed)
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
        tf.random.set_seed(self.seed)
        if self.keep_actor_data:
            dataset_shape = (tf.TensorShape([None, 25]), tf.TensorShape([]), tf.TensorShape([2, 25]))
        else:
            dataset_shape = (tf.TensorShape([None, 25]), tf.TensorShape([]))
        return self.train_dataset \
            .shuffle(buffer_size=shuffle_buffer_size) \
            .map(map_func, num_parallel_calls=num_parallel_calls) \
            .padded_batch(batch_size, dataset_shape) \
            .prefetch(prefetch)

    def val_iterator(self, batch_size: int = 32, prefetch: int = 3, num_parallel_calls: int = -1) -> tf.data.Dataset:
        self.assert_if_dataset_is_not_none(self.val_dataset)
        return self._get_evaluation_set_iterator(self.val_dataset, batch_size, prefetch, num_parallel_calls)

    def test_iterator(self, batch_size: int = 32, prefetch: int = 3, num_parallel_calls: int = -1) -> tf.data.Dataset:
        self.assert_if_dataset_is_not_none(self.test_dataset)
        return self._get_evaluation_set_iterator(self.test_dataset, batch_size, prefetch, num_parallel_calls)

    def _get_evaluation_set_iterator(self, dataset: tf.data.Dataset, batch_size: int = 32,
                                     prefetch: int = 3, num_parallel_calls: int = -1) -> tf.data.Dataset:
        map_func = self.get_map_func()
        if self.keep_actor_data:
            dataset_shape = (tf.TensorShape([None, 25]), tf.TensorShape([]), tf.TensorShape([2, 25]))
        else:
            dataset_shape = (tf.TensorShape([None, 25]), tf.TensorShape([]))
        return dataset \
            .map(map_func, num_parallel_calls=num_parallel_calls) \
            .padded_batch(batch_size, dataset_shape) \
            .prefetch(prefetch)

    def get_map_func(self) -> Callable:
        audio_func = self.get_audio_func()
        if self.keep_actor_data:
            return lambda *items: tf.py_function(func=audio_func, inp=items, Tout=[tf.float64, tf.int32, tf.float64])
        return lambda *items: tf.py_function(func=audio_func, inp=items, Tout=[tf.float64, tf.int32])

    def get_audio_func(self) -> Callable:
        if self.data_status == 'raw_data':
            return self._load_audio_raw
        elif 'wav2vec' in self.data_status:
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

    def _load_audio_wav2vec(self, file_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        audio = np.load(bytes.decode(file_path.numpy()))[0]
        if self.padding_value is None or self.total_length is None:
            return tf.convert_to_tensor(audio, dtype=tf.float64), label
        else:
            temp = tf.convert_to_tensor(audio, dtype=tf.float64)
            wav2vec_out = temp[..., :self.total_length, :]
            if len(wav2vec_out.shape) == 3:
                wav2vec_out = tf.concat((wav2vec_out, tf.experimental.numpy.full(
                                             (wav2vec_out.shape[0], self.total_length - wav2vec_out.shape[1],
                                              wav2vec_out.shape[-1]),
                                             fill_value=self.padding_value)), axis=1)
            else:
                wav2vec_out = tf.concat((wav2vec_out, tf.experimental.numpy.full(
                                             (wav2vec_out.shape[0], self.total_length - wav2vec_out.shape[1]),
                                             fill_value=self.padding_value)), axis=1)
            return wav2vec_out, label

    def _load_numpy_features(self, file_path: tf.Tensor, label: tf.Tensor, k: tf.Tensor = tf.constant([1.], dtype=tf.float64))\
            -> Union[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        data = np.load(bytes.decode(file_path.numpy()))
        if self.crop:
            data = data[:self.number_of_windows]
        elif self.total_length is not None and self.padding_value is not None:
            data = data[0]
            data = np.transpose(data)
            data = data[..., :self.total_length, :]
            data = np.concatenate(
                (data, np.full((self.total_length - data.shape[0], data.shape[1]), fill_value=self.padding_value)), axis=0)
        if self.keep_actor_data:
            return tf.convert_to_tensor(data, dtype=tf.float64), label, k
        return tf.convert_to_tensor(data, dtype=tf.float64), label

    def get_numpy_dataset(self, dataset: tf.data.Dataset) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, Tuple]]:
        self.assert_if_dataset_is_not_none(dataset)

        numpy_dataset = np.array(list(dataset.map(self.get_map_func()).as_numpy_iterator()), dtype=object)
        if self.keep_actor_data:
            return np.stack(numpy_dataset[:, 0]).squeeze(), \
                   (np.stack(numpy_dataset[:, 1])[:, 0].astype("int32").squeeze(),
                        np.stack(numpy_dataset[:, 1])[:, 1].astype("int32").squeeze())
        return np.stack(numpy_dataset[:, 0]).squeeze(), numpy_dataset[:, 1].astype("int32").squeeze()

    def create_dataset_with_statistics(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        numpy_dataset = np.array(list(dataset.map(self.get_map_func()).as_numpy_iterator()), dtype=object)
        stats = self.compute_ground_truth_per_actor_mean_and_std(numpy_dataset[:, 0],
                                                       np.stack(numpy_dataset[:, 1])[:, 1].astype("int32").squeeze())
        means_stds_list = [stats[actor_label] for actor_label in np.stack(numpy_dataset[:, 1])[:, 1]]
        means_stds_dataset = tf.data.Dataset.from_tensor_slices(means_stds_list)
        return tf.data.Dataset.zip((dataset.map(lambda x, y: x),
                                    dataset.map(lambda a, b: b[0]), means_stds_dataset))

    @staticmethod
    def compute_ground_truth_per_actor_mean_and_std(data: np.ndarray, actor_labels: np.ndarray) -> dict:
        result = dict.fromkeys(set(actor_labels))
        for actor_label in set(actor_labels):
            actor_data = [sample for sample, actor_id in zip(data, actor_labels) if actor_id == actor_label]
            concatenated_actor_data = np.concatenate(actor_data, axis=0)
            result[actor_label] = np.mean(concatenated_actor_data, axis=0), np.std(concatenated_actor_data, axis=0)
        return result

    @staticmethod
    def assert_if_dataset_is_not_none(dataset: tf.data.Dataset) -> None:
        assert dataset, 'Dataset has to be initialized first.'
