from datasets import RavdessReader
from preprocessors.preprocessor import Preprocessor
from preprocessors.wav2vec import Wav2VecModel

import tensorflow as tf
import numpy as np

class Wav2VecPreprocessor(Preprocessor):
    def __init__(self, dataset, target_dir, wav2vec_model):
        super().__init__(dataset, target_dir)
        self.wav2vec = wav2vec_model

    def preprocess_single_example(self, example):
        return self.wav2vec(tf.expand_dims(example, axis=0))

    def save_single_example(self, target_path, preprocessed_example):
        np.save(target_path, preprocessed_example.numpy())


if __name__ == '__main__':
    dataset = RavdessReader(desired_sampling_rate=16000,
                            total_length=80000,
                            padding_value=0.0,
                            train_size=0.0,
                            val_size=1.0)  # because val is not shuffled
    preprocessor = Wav2VecPreprocessor(dataset, "wav2vec_large_data_cnn", Wav2VecModel(dataset.number_of_classes, agg=None, model='large'))
    preprocessor.preprocess_data()
