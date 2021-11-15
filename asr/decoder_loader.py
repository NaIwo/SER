import torch


class DecoderLoader:
    def __init__(self, language: str = 'en'):
        self.language = language
        self.repo = 'snakers4/silero-models'
        self.model = 'silero_stt'

    def get_decoder(self):
        _, decoder, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-models', model='silero_stt', language=self.language)
        return decoder, utils
