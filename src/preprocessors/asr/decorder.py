import torch


class DecoderLoader:
    def __init__(self, model: str, repo: str, language: str):
        self.language = language
        self.repo = repo
        self.model = model

    def get_decoder(self):
        _, decoder, utils = torch.hub.load(
            repo_or_dir=self.repo, model=self.model, language=self.language)
        return decoder, utils
