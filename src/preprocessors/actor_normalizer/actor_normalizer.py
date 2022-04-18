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
        return self.dataset.get_audio_func()(path, y)

    def preprocess_data(self) -> None:
        actors: List = self.dataset.data_labels.actors
        actors_data: DefaultDict = defaultdict(lambda: defaultdict(list))
        actor: int
        for (path, y), actor in zip(self.dataset.full_dataset, actors):
            data, _ = self._load_audio(path, y)

            actors_data[actor]['path'].append(path)
            actors_data[actor]['data'].append(data.numpy())

        for actor in actors_data.keys():
            with tf.device('/CPU:0'):
                data = tf.ragged.stack(actors_data[actor]['data']).to_tensor(np.nan).numpy()
            single_sample_shape = data.shape[1:]
            data = data.squeeze()
            if (dim_number := len(data.shape)) <= 3:
                data = (data - np.nanmean(data, (tuple(range(dim_number - 1))))) \
                        / (np.nanstd(data, tuple(range(dim_number - 1))) + 1e-10)
            else:
                raise Exception("Arrays of dimension higher than 3 are not supported")
            for idx, features in enumerate(data):
                path = actors_data[actor]['path'][idx]

                raw_path_string = bytes.decode(path.numpy())
                raw_path_string = raw_path_string.replace(self.dataset.data_status, self.target_directory)

                path_to_save = raw_path_string.replace(f'.{config["data"]["path-extension"]}', '')
                dir_to_save = os.sep.join(path_to_save.split(os.sep)[:-1])

                if not (os.path.exists(dir_to_save)):
                    Path(dir_to_save).mkdir(parents=True, exist_ok=True)

                features = features[~np.isnan(features).all(axis=-1)]
                for axis in (0, -1):  # additional dim can be leading or trailing only
                    if single_sample_shape[axis] == 1:
                        features = np.expand_dims(features, axis=axis)
                self.save_single_example(path_to_save, features)

    def save_single_example(self, target_path: str, preprocessed_example: tf.Tensor):
        np.save(target_path, preprocessed_example)


if __name__ == '__main__':
    Dataset = get_dataset_by_name(config['data']['dataset']['name'])
    dataset = Dataset(data_status='mfcc',
                      total_length=None,
                      padding_value=0.0,
                      resample_training_set=False
                      )
    preprocessor = Normalizer(dataset, config['data']['out-name'])
    preprocessor.preprocess_data()
