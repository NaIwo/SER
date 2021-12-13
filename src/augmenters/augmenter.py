import os
from abc import abstractmethod
from pathlib import Path
import numpy as np
import tensorflow as tf
import soundfile as sf
from typing import Union, List
from collections import Counter


from src.datasets import BaseDataset


class Augmenter:
    def __init__(self, dataset: BaseDataset, target_dir: str, sampling_rate: int = 44100):  # 44100 is the most common sr
        self.dataset = dataset
        self.target_directory = target_dir
        self.sampling_rate = sampling_rate

    def _load_audio(self, path, y):
        return self.dataset._load_audio_raw(path, y)

    def augment_data(self, balance=False) -> None:
        class_ratios = self._compute_class_ratios() if balance else None
        for path, y in self.dataset.train_dataset:
            data, label = self._load_audio(path, y)
            raw_path_string = bytes.decode(path.numpy())
            path_to_save = raw_path_string.replace('raw_data', self.target_directory)

            dir_to_save = os.sep.join(path_to_save.split(os.sep)[:-1])

            number_of_generated_examples = class_ratios[y.numpy()]
            if number_of_generated_examples > 0:
                augmented_examples = self.augment_example(data, number_of_generated_examples)
            else:
                augmented_examples = None

            if not (os.path.exists(dir_to_save)):
                Path(dir_to_save).mkdir(parents=True, exist_ok=True)

            if number_of_generated_examples == 1:
                if balance:
                    self.save_generated_examples(path_to_save, list(augmented_examples))
                else:
                    self.save_generated_example(path_to_save, augmented_examples)
            elif number_of_generated_examples > 1:
                self.save_generated_examples(path_to_save, augmented_examples)

    def _compute_class_ratios(self):
        labels = [label for _, label in list(self.dataset.train_dataset.as_numpy_iterator())]
        label_counter = Counter(labels)
        max_count = max(label_counter.values())
        for label in label_counter:
            label_counter[label] = max_count // label_counter[label] - 1  # rough balancing, we won't get exactly same number of examples per class
        return label_counter

    @abstractmethod
    def augment_example(self, example: tf.Tensor, number_of_generated_examples=1) -> Union[List[np.ndarray], np.ndarray]:
        pass

    def save_generated_examples(self, target_path: str, augmented_examples: List[np.ndarray]):
        for i, augmented_example in enumerate(augmented_examples):
            target_path_for_example = self._generate_target_path_for_example(target_path, i)
            self.save_generated_example(target_path_for_example, augmented_example)

    def _generate_target_path_for_example(self, base_path: str, number: int):
        last_dot_index = base_path.rfind('.')
        part_before_extension, part_after_extension = base_path[:last_dot_index], base_path[last_dot_index:]
        return f'{part_before_extension}_a{number}{part_after_extension}'

    def save_generated_example(self, target_path: str, augmented_example: np.ndarray) -> None:
        sf.write(target_path, augmented_example, self.sampling_rate)
