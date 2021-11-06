from transformers import AutoConfig, TFWav2Vec2Model
import tensorflow as tf


class Wav2VecModel:
    def __init__(self, num_of_classes):
        self.wav2vec_name = "facebook/wav2vec2-base"

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
        self.wav2vec = TFWav2Vec2Model(self.config)

    def __call__(self, inputs):
        wav2vec_out = self.wav2vec(inputs, training=False)
        return tf.math.reduce_mean(wav2vec_out.last_hidden_state, axis=1)