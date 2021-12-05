import os
from abc import abstractmethod
from pathlib import Path
import numpy as np
import tensorflow as tf
from typing import Union, Callable, Optional

from src.datasets import BaseDataset
from src.config_reader import config


class Preprocessor:
    def __init__(self, dataset: BaseDataset, target_dir: str, reduce_func: Optional[Callable] = None):
        self.dataset = dataset
        self.target_directory = target_dir
        self.agg = reduce_func

    def _load_audio(self, path, y):
        return self.dataset._load_audio_raw(path, y)

    def preprocess_data(self) -> None:
        for path_iterator in [self.dataset.test_dataset]:
            a = os.listdir('/home/iwo/Pulpit/Studia/SER/src/datasets/datasets_files/MELD/960wav2vec_mean_normalized/test/output_repeated_splits_test')
            for path, y in path_iterator:
                if bytes.decode(path.numpy()).split('/')[-1].replace('.wav', '.npy') in a:
                    continue
                data, _ = self._load_audio(path, y)
                raw_path_string = bytes.decode(path.numpy())
                raw_path_string = raw_path_string.replace('raw_data', self.target_directory)

                path_to_save = raw_path_string.replace(f'.{config["data"]["path-extension"]}', '')
                dir_to_save = os.sep.join(path_to_save.split(os.sep)[:-1])

                features = self.preprocess_single_example(data)

                if not (os.path.exists(dir_to_save)):
                    Path(dir_to_save).mkdir(parents=True, exist_ok=True)

                self.save_single_example(path_to_save, features)

    def preprocess_batch(self, batch: tf.Tensor):  # NOT TESTED
        return tf.map_fn(self.preprocess_single_example, batch)

    @abstractmethod
    def preprocess_single_example(self, example: tf.Tensor) -> Union[tf.Tensor, np.ndarray]:
        pass

    @abstractmethod
    def save_single_example(self, target_path: str, preprocessed_example: Union[tf.Tensor, np.ndarray]) -> None:
        pass
