import tensorflow_hub as tf_hub


class ModelLoader:
    def __init__(self, language: str = 'en'):
        self.language = language
        self.model_url = 'https://tfhub.dev/silero/silero-stt/en/1'

    def load_model(self):
        return tf_hub.load(self.model_url)
