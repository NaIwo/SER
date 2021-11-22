import tensorflow_hub as tf_hub


class ModelLoader:
    def __init__(self, model_url, language: str = 'en'):
        self.language = language
        self.model_url = model_url

    def load_model(self):
        return tf_hub.load(self.model_url)
