import numpy as np
import tensorflow as tf

from typing import Tuple, Union, List
from nlpaug.augmenter.audio import vtlp

from src.augmenters.augmenter import Augmenter
from src.config_reader import config
from src.datasets import BaseDataset, get_dataset_by_name


class VtlpAugmenter(Augmenter):

    def __init__(self, dataset: BaseDataset, target_dir: str, modification_range: Tuple[float, float], coverage: float = 1.0,
                 frequency_threshold: int = 4800):
        super().__init__(dataset, target_dir)
        self.vtlp = vtlp.VtlpAug(self.sampling_rate, modification_range, coverage, frequency_threshold)

    def augment_example(self, example: tf.Tensor, number_of_generated_examples=1) -> Union[List[np.ndarray], np.ndarray]:
        return self.vtlp.augment(example.numpy(), number_of_generated_examples)


def main():
    dataset_name = config['data']['dataset']['name']
    Dataset = get_dataset_by_name(dataset_name)
    dataset = Dataset(desired_sampling_rate=None,  # we want native sr
                      total_length=None,
                      padding_value=None,  # we don't want any padding for augmentation
                      train_size=1.0,
                      test_size=0.0,
                      val_size=0.0,
                      data_status='raw_data')
    vtlp_props = config['augmenters']['vltp']
    augmenter = VtlpAugmenter(dataset, config["data"]["augmented-name"], (vtlp_props['modification_start'], vtlp_props['modification_end']),
                              vtlp_props['coverage'], vtlp_props['frequency_threshold'])
    if dataset_name == "MELD":
        augmenter.augment_data(balance=True)
    elif dataset_name == "RAVDESS":
        augmenter.augment_data()


if __name__ == '__main__':
    main()
