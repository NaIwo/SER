from src.preprocessors.config_reader import config
from src.datasets import get_dataset_by_name
from src.preprocessors.preprocessor import Preprocessor
from src.datasets import BaseDataset

import tensorflow as tf
import numpy as np
from pathlib import Path
from collections import defaultdict
import os
from typing import List, DefaultDict
import warnings


class Normalizer(Preprocessor):
    def __init__(self, dataset: BaseDataset, target_dir: str):
        super().__init__(dataset, target_dir)
        warnings.warn(
            'This only applies to Ravdess Dataset! If you want to use it on different dataset, provide new implementation.')

    def _load_audio(self, path, y):
        return self.dataset._load_audio_wav2vec(path, y)

    def preprocess_data(self) -> None:
        actors: List = self.dataset.data_labels.actors
        actors_data: DefaultDict = defaultdict(lambda: defaultdict(list))
        actor: int
        for (path, y), actor in zip(self.dataset.full_dataset, actors):
            data, _ = self._load_audio(path, y)

            actors_data[actor]['path'].append(path)
            actors_data[actor]['data'].append(data.numpy())

        for actor in actors_data.keys():
            data = np.array(actors_data[actor]['data'])
            data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-10)
            for idx, features in enumerate(data):
                path = actors_data[actor]['path'][idx]

                raw_path_string = bytes.decode(path.numpy())
                raw_path_string = raw_path_string.replace('960wav2vec_sum_normalized', self.target_directory)

                path_to_save = raw_path_string.replace(f'.{config["data"]["path-extension"]}', '')
                dir_to_save = os.sep.join(path_to_save.split(os.sep)[:-1])

                if not (os.path.exists(dir_to_save)):
                    Path(dir_to_save).mkdir(parents=True, exist_ok=True)

                self.save_single_example(path_to_save, features[np.newaxis, ...])

    def save_single_example(self, target_path: str, preprocessed_example: tf.Tensor):
        np.save(target_path, preprocessed_example)


if __name__ == '__main__':
    Dataset = get_dataset_by_name(config['data']['dataset']['name'])
    dataset = Dataset(data_status='960wav2vec_sum_normalized',
                      total_length=None,
                      padding_value=0.0,
                      resample_training_set=False
                      )
    preprocessor = Normalizer(dataset, config['data']['out-name'])
    preprocessor.preprocess_data()
