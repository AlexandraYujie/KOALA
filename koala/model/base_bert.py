import importlib
from unicodedata import name
import torch.nn as nn
import transformers
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer, AutoModel, AutoConfig
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers import XLMRobertaModel, XLMRobertaConfig
from transformers import ElectraModel, ElectraPreTrainedModel
from transformers import DebertaV2Model, DebertaV2PreTrainedModel
import torch

class XLMRobertaPreTrainedModel(RobertaPreTrainedModel):
    """
    This class overrides [`RobertaModel`]. Please check the superclass for the appropriate documentation alongside
    usage examples.
    """

    config_class = XLMRobertaConfig


base_class_mapping={
    "roberta-base": RobertaPreTrainedModel,
    "google/electra-base-discriminator": ElectraPreTrainedModel,
    "xlm-roberta-base": XLMRobertaPreTrainedModel,
    "xlm-roberta-large": XLMRobertaPreTrainedModel,
    "bert-base-uncased": BertPreTrainedModel,
    "bert-large-uncased": BertPreTrainedModel,
    "microsoft/mdeberta-v3-base": DebertaV2PreTrainedModel,
    "bert-base-multilingual-uncased": BertPreTrainedModel

}

model_object_mapping = {
    "roberta-base": RobertaModel,
    "google/electra-base-discriminator": ElectraModel,
    "xlm-roberta-base": XLMRobertaModel,
    "xlm-roberta-large": XLMRobertaModel,
    "bert-base-uncased": BertModel,
    "bert-large-uncased": BertModel,
    "microsoft/mdeberta-v3-base": DebertaV2Model,
    "bert-base-multilingual-uncased": BertModel

}


transformers_module = dir(transformers)

def find_class_names(model_type, class_type):
    model_type = model_type.replace("-", "").lower()
    for item in transformers_module:
        if model_type + class_type == item.lower():
            return item

    return None


def class_factory(name_or_path):
    loadedConfig  = AutoConfig.from_pretrained(name_or_path)  # loadedconfig of nam
    model_type = loadedConfig.model_type  # loadedconfig of model type
    pretrained_class = find_class_names(model_type, 'pretrainedmodel')
    model_class = find_class_names(model_type, 'model')

    if pretrained_class is not None:
        pretrained_class_object = getattr(transformers, pretrained_class)
    elif model_type == 'xlm-roberta':
        pretrained_class_object = XLMRobertaPreTrainedModel
    elif base_class_mapping.get(name_or_path) is not None:
        pretrained_class_object = base_class_mapping.get(name_or_path)
    else:
        raise ValueError("Could not find correct pretrained class for the model type {model_type} in transformers library")

    if model_class != None:
        model_class_object = getattr(transformers, model_class)  # 先到transformer里面找一下这个类
    elif model_object_mapping.get(name_or_path) is not None:
        model_class_object = model_object_mapping.get(name_or_path)  # 否则就到字典里面去找
    else:
        raise ValueError("Could not find correct model class for the model type {model_type} in transformers library")

    class BaseBERT(pretrained_class_object):
        """
            Shallow wrapper around HuggingFace transformers. All new parameters should be defined at this level.

            This makes sure `{from,save}_pretrained` and `init_weights` are applied to new parameters correctly.
        """
        _keys_to_ignore_on_load_unexpected = [r"cls"]

        def __init__(self, config, base_config):
            super().__init__(config)  # config 是 model config & base_config 是被model config 更新过以后的参数（setting里面自定义的config）
            self.config = config
            self.dim = base_config.dim
            self.linear = nn.Linear(config.hidden_size, base_config.dim, bias=False)
            self.linear_to_1 = nn.Sequential(
                nn.Linear(config.hidden_size, base_config.dim),
                nn.ReLU(),
                nn.Linear(base_config.dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            setattr(self, self.base_model_prefix, model_class_object(config))

            # if base_config.relu:
            #     self.score_scaler = nn.Linear(1, 1)

            self.init_weights()

            # if base_config.relu:
            #     self.score_scaler.weight.data.fill_(1.0)
            #     self.score_scaler.bias.data.fill_(-8.0)

        @property
        def LM(self):
            base_model_prefix = getattr(self, "base_model_prefix")
            return getattr(self, base_model_prefix)

        @classmethod
        def from_pretrained(cls, name_or_path, base_config):
            # if name_or_path.endswith('.dnn'):
            #     dnn = torch_load_dnn(name_or_path)
            #     base = dnn.get('arguments', {}).get('model', 'bert-base-uncased')
            #
            #     obj = super().from_pretrained(base, state_dict=dnn['model_state_dict'], base_config=base_config)
            #     obj.base = base
            #
            #     return obj

            obj = super().from_pretrained(name_or_path, base_config=base_config)
            obj.base = name_or_path

            return obj

        @staticmethod
        def raw_tokenizer_from_pretrained(name_or_path):
            obj = AutoTokenizer.from_pretrained(name_or_path)
            obj.base = name_or_path

            return obj

    return BaseBERT


if __name__ == "__main__":
    class_factory("bert-base-chinese")
