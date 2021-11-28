from transformers import AutoConfig, TFWav2Vec2Model, Wav2Vec2Processor
import tensorflow as tf

from src.config_reader import config


class Wav2VecModel:
    def __init__(self, num_of_classes):

        self.wav2vec_name = config['model']['wav2vec2']['pretrained-model']
        self.wav2vec_preprocessor_name = config['model']['wav2vec2']['preprocessor']

        self.agg = config['model']['wav2vec2']['aggregation']

        self.sampling_rate = config['model']['wav2vec2']['desired-sampling-rate']
        self.padding_value = config['data']['dataset']['padding-value']

        self.normalize: bool = config['model']['wav2vec2']['normalize']

        self.config = AutoConfig.from_pretrained(
            self.wav2vec_name,
            num_labels=num_of_classes,
            finetuning_task="wav2vec2_clf",
            problem_type='single_label_classification'
        )

        # Wav2Vec2 models that have set config.feat_extract_norm == "group",
        # (in our case that situation is taking place)
        # such as wav2vec_neural-base, have not been trained using attention_mask.
        # For such models, input_values should simply be padded with 0 and no attention_mask should be passed.
        self.wav2vec = TFWav2Vec2Model.from_pretrained(self.wav2vec_name, config=self.config)
        self.processor = Wav2Vec2Processor.from_pretrained(self.wav2vec_preprocessor_name, config=self.config)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        wav2vec_out = self.processor(inputs, sampling_rate=self.sampling_rate, padding_value=self.padding_value,
                                     return_tensors='tf', do_normalize=self.normalize)
        wav2vec_out = tf.squeeze(wav2vec_out.input_values, axis=1)
        wav2vec_out = self.wav2vec(wav2vec_out, training=False)
        return self._get_final_results(wav2vec_out.last_hidden_state)

    def _get_final_results(self, results: tf.Tensor) -> tf.Tensor:
        if self.agg == 'mean':
            results = tf.math.reduce_mean(results, axis=1)
        elif self.agg == 'sum':
            results = tf.math.reduce_sum(results, axis=1)

        return results
