import os
from abc import abstractmethod
from pathlib import Path

import numpy as np

from datasets import DatasetReaderBase
import tensorflow as tf
from typing import Union


class Preprocessor:
    def __init__(self, dataset: DatasetReaderBase, target_dir: str):
        self.dataset = dataset
        self.target_directory = target_dir

    def preprocess_data(self) -> None:
        path_iterator = self.dataset.test_dataset

        for path, y in path_iterator:
            data, _ = self.dataset._load_audio_raw(path, y)
            raw_path_string = bytes.decode(path.numpy())
            raw_path_string = raw_path_string.replace('raw_data', self.target_directory)

            path_to_save = raw_path_string.replace('.wav', '')
            dir_to_save = os.sep.join(path_to_save.split(os.sep)[:-1])

            features = self.preprocess_single_example(data)

            if not (os.path.exists(dir_to_save)):
                Path(dir_to_save).mkdir(parents=True, exist_ok=True)
            self.save_single_example(path_to_save, features)

    @abstractmethod
    def preprocess_single_example(self, example: tf.Tensor) -> Union[tf.Tensor, np.ndarray]:
        pass

    @abstractmethod
    def save_single_example(self, target_path: str, preprocessed_example: Union[tf.Tensor, np.ndarray]) -> None:
        pass
