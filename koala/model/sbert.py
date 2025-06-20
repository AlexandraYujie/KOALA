import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
import torch.nn.functional as F
from typing import List, Union
from torch import Tensor
from numpy import ndarray
import numpy as np
from tqdm.autonotebook import trange
import torch.nn as nn
import os
import json
from koala.model.utils import weights_init


class SbertBase(SentenceTransformer):
    def __init__(self, config):
        super().__init__(config.checkpoint, device=config.device)
        self.config = config
        self.hidden_size = self._get_hidden_size()
        self.unused_ids = self._get_unused_ids() * 2
        print(f"hidden_size of {config.checkpoint} is {self.hidden_size}...")


    def _get_unused_ids(self):
        ks = self.tokenizer.get_vocab().keys()
        unused_list = []
        for k in ks:
            if "unused" in k:
                unused_list.append(k)
        unused_list = sorted(unused_list, key=lambda x: eval(x.replace("unused", "")[1:][:-1]), reverse=False)
        unused_ids = []
        for unused_word in unused_list:
            unused_ids.append(self.tokenizer.convert_tokens_to_ids(unused_word))
        print("length of unused ids:",len(unused_ids))
        return unused_ids

    def _get_hidden_size(self):
        transformer_json_path = os.path.join(self.config.checkpoint, 'config.json')
        if not os.path.exists(transformer_json_path):
            raise RuntimeError(f"No {transformer_json_path} file.")
        with open(transformer_json_path) as fp:
            transformer_config = json.load(fp)
            hidden_size = transformer_config.get("hidden_size", None)
            if hidden_size == None:
                raise RuntimeError(f"No hidden_size parameter in {transformer_json_path}.")
            return hidden_size


    def encode(self, sentences: Union[str, List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False,
               use_unused_ids: bool = False) -> Union[List[Tensor], ndarray, Tensor]:



        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != 'sentence_embedding':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self._target_device

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            # if not use_unused_ids:
            features = self.tokenize(sentences_batch)

            if use_unused_ids:
                features["input_ids"] = torch.tensor(np.array([[sentences[0]] + self.unused_ids[:(len(sentences)-2)] + [sentences[-1]] for sentences in list(features["input_ids"].numpy())]))
                # print(features["input_ids"].size())

            features = batch_to_device(features, device)
            out_features = self.forward(features)

            if output_value == 'token_embeddings':
                embeddings = []
                for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
                    last_mask_id = len(attention)-1
                    while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                        last_mask_id -= 1

                    embeddings.append(token_emb[0:last_mask_id+1])
            elif output_value is None:  #Return all outputs
                embeddings = []
                for sent_idx in range(len(out_features['sentence_embedding'])):
                    row =  {name: out_features[name][sent_idx] for name in out_features}
                    embeddings.append(row)
            else:   #Sentence embeddings
                embeddings = out_features[output_value]
                embeddings = embeddings
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def forward(self, input):  # 改写forward
        for module in self:
            input = module(input)
        return input

class Sbert(torch.nn.Module):
    def __init__(self, config, pair_mode=True):
        super(Sbert, self).__init__()
        # load hidden_size of bert
        self.config = config
        self.pair_mode = pair_mode
        self.random_query = config.random_query
        self.model = SbertBase(config)
        self.similarity = config.similarity
        if self.config.no_query:  # 如果开启了no_query training mode
            self.linear_to_1 = nn.Sequential(
                nn.Linear(self.model.hidden_size, self.config.dim),
                nn.ReLU(),
                nn.Linear(self.config.dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        else:  # 增加参数
            self.add_linear = nn.Sequential(
                nn.Linear(self.model.hidden_size, self.model.hidden_size),
                # nn.Dropout(0.1),
                nn.LayerNorm(self.model.hidden_size),
                nn.ReLU(),
                nn.Linear(self.model.hidden_size, self.model.hidden_size),
            )
            self.add_linear.apply(weights_init)


    def forward(self, Q, D):  # Q length = 4, D length = 8
        # Q = self.query(*Q)  # torch.Size([4, 32, 128])
        if Q != None:
            Q_repr = self.model.encode(Q, convert_to_tensor=True, device=self.model.device, show_progress_bar=False, use_unused_ids=self.random_query)
            D_repr = self.model.encode(D, convert_to_tensor=True, device=self.model.device, show_progress_bar=False)
            if not self.config.no_linear_add:
                Q_repr = self.add_linear(Q_repr)
                D_repr = self.add_linear(D_repr)
            if self.pair_mode:
                Q_repr = Q_repr.repeat_interleave(2, dim=0).contiguous()  # ([256, 32, 128])
            scores = self.get_score(Q_repr, D_repr)
            # scores = torch.sum(Q_repr * D_repr, dim=1)
        else:  # if do not have Q
            D_repr = self.model.encode(D, convert_to_tensor=True, device=self.model.device, show_progress_bar=False)
            scores = self.linear_to_1(D_repr)
        return scores


    def get_score(self, Q_repr, D_repr):
        similarity_method = self.similarity
        if similarity_method == "cos":
            scores = F.cosine_similarity(Q_repr, D_repr, dim=1)
        elif similarity_method == "dot":
            # Dot product
            scores = torch.sum(Q_repr * D_repr, dim=1)
        elif similarity_method == "euc":
            # Euclidean distance
            scores = -torch.norm(Q_repr - D_repr, p=2, dim=1)
        elif similarity_method == "manh":
            # Manhattan distance
            scores = -torch.sum(torch.abs(Q_repr - D_repr), dim=1)

        return scores

if __name__ == "__main__":
    print(torch.cuda.is_available())
    model = SentenceTransformer("../all-MiniLM-L6-v2", device="cuda")
    model.to(torch.device("cuda"))
    print(model.device)