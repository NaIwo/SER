import tensorflow as tf
import torch
import numpy as np

from src.config_reader import config
from src.datasets import BaseDataset, get_dataset_by_name
from src.preprocessors.preprocessor import Preprocessor
from src.preprocessors.asr.model import ModelLoader
from src.preprocessors.asr.decorder import DecoderLoader


class AsrPreprocessor(Preprocessor):
    def __init__(self, dataset: BaseDataset, target_dir: str, model: tf.Model, decoder, decoder_utils: tuple):
        super().__init__(dataset, target_dir)
        self.model = model
        self.decoder = decoder

        read_batch, split_into_batches, read_audio, prepare_model_input = decoder_utils
        self.read_batch = read_batch
        self.split_into_batches = split_into_batches
        self.read_audio = read_audio
        self.prepare_model_input = prepare_model_input

    def preprocess_single_example(self, example):
        res = self.model(tf.constant(tf.cast(example, dtype='float32'))[0])['output_0']
        text = self.decoder(torch.Tensor(res.numpy()))
        return text

    def save_single_example(self, target_path: str, preprocessed_example: str):
        with open(target_path, 'w') as f:
            f.write(preprocessed_example)


if __name__ == '__main__':
    Dataset = get_dataset_by_name(config['data']['dataset']['name'])
    dataset = Dataset(desired_sampling_rate=config['data']['dataset']['desired-sampling-rate'],
                      total_length=config['data']['dataset']['desired-length'],
                      padding_value=config['data']['dataset']['padding-value'],
                      train_size=config['data']['dataset']['train-size'],
                      test_size=config['data']['dataset']['test-size'],
                      val_size=config['data']['dataset']['val-size'],
                      data_status='raw_data',
                      train_test_seed=config['data']['dataset']['shuffle-seed'])
    model = ModelLoader(config['model']['asr']['tfhub'])
    decoder_loader = DecoderLoader(model=config['model']['asr']['decoder']['model'],
                                   repo=config['model']['asr']['decoder']['repo'],
                                   language=config['model']['asr']['decoder']['language'])
    decoder, utils = decoder_loader.get_decoder()
    preprocessor = AsrPreprocessor(dataset, 'test', model.signatures["serving_default"], decoder, utils)
    preprocessor.preprocess_data()
