from transformers import AutoConfig, TFWav2Vec2Model, Wav2Vec2Processor
import tensorflow as tf

from src.config_reader import config


class Wav2VecModel:
    def __init__(self, num_of_classes):

        self.wav2vec_name = config['model']['wav2vec2']['pretrained-model']
        self.wav2vec_preprocessor_name = config['model']['wav2vec2']['preprocessor']

        self.sampling_rate = config['model']['wav2vec2']['desired-sampling-rate']
        self.padding_value = config['data']['dataset']['padding-value']

        self.normalize: bool = config['model']['wav2vec2']['normalize']

        self.config = AutoConfig.from_pretrained(
            self.wav2vec_name,
            num_labels=num_of_classes,
            finetuning_task="wav2vec2_clf",
            problem_type='single_label_classification',
            output_hidden_states=True
        )

        # Wav2Vec2 models that have set config.feat_extract_norm == "group",
        # (in our case that situation is taking place)
        # such as wav2vec_neural-base, have not been trained using attention_mask.
        # For such models, input_values should simply be padded with 0 and no attention_mask should be passed.
        self.wav2vec = TFWav2Vec2Model.from_pretrained(self.wav2vec_name, config=self.config)
        self.processor = Wav2Vec2Processor.from_pretrained(self.wav2vec_preprocessor_name, config=self.config)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        wav2vec_out = self.processor(inputs, sampling_rate=16000, padding_value=self.padding_value,
                                     return_tensors='tf', do_normalize=self.normalize)
        wav2vec_out = tf.squeeze(wav2vec_out.input_values, axis=1)
        wav2vec_out = self.wav2vec(wav2vec_out, training=False)
        wav2vec_out = tf.stack(wav2vec_out.hidden_states, axis=1)
        wav2vec_out = wav2vec_out[:, :, :250, :]
        #wav2vec_out = tf.concat((wav2vec_out, tf.zeros((1, 13, 250 - wav2vec_out.shape[2], wav2vec_out.shape[-1]))), axis=2)
        return wav2vec_out

