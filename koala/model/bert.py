import os
import torch
import sys
from transformers import AutoTokenizer
from koala.model.base_bert import class_factory
# from config.config import Config


class BERT(torch.nn.Module):
    """
    Shallow module that wraps the ColBERT parameters, custom configuration, and underlying tokenizer.
    This class provides direct instantiation and saving of the model/colbert_config/tokenizer package.

    Like HF, evaluation mode is the default.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config  # Config.from_existing(Config.load_from_checkpoint(config.checkpoint), config)  # 前者是None，后者是config;
        # 前者若不是none，前者的参数替代后者
        self.name = self.config.checkpoint
        base_bert = class_factory(self.name)
        self.model = base_bert.from_pretrained(self.name, base_config=self.config)
        self.raw_tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.eval()

    @property
    def device(self):
        return self.model.device

    @property
    def bert(self):
        return self.model.LM

    @property
    def linear(self):
        return self.model.linear

    @property
    def linear_to_1(self):
        return self.model.linear_to_1

    @property
    def score_scaler(self):
        return self.model.score_scaler

    def save(self, path):
        # self.model.save_pretrained(path)  # save the model(BERT, no mlp?)
        # self.raw_tokenizer.save_pretrained(path)  # save tokenizer
        self.config.save_for_checkpoint(path)  # save config