from koala.config.config import Config
from koala.model.bert import BERT
import torch.nn as nn
import torch
import string

class HiddenBERT(BERT):
    def __init__(self, config, **args):
        super().__init__(config)
        self.config = config
        # if self.config.mask_punctuation:
        #     self.skiplist = {w: True
        #                      for symbol in string.punctuation
        #                      for w in [symbol, self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]}
        self.pad_token = self.raw_tokenizer.pad_token_id
        self.sigmoid_score = config.sigmoid_score


    def forward(self, info, info_mask):
        # Q = self.query(*Q)  # torch.Size([4, 32, 128])
        info = self.encode_info(info, info_mask, keep_dims=True)  # torch.Size([8, 220, 128]) torch.Size([8, 220, 1])
        # scores = self.score(Q_duplicated, D, D_mask)  # 前两个size有问题
        scores = self.score(info)
        if self.sigmoid_score:
            scores = torch.sigmoid(scores)
        return scores  # torch.Size([8])

    # def query(self, input_ids, attention_mask):
    #     input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
    #     Q = self.bert(input_ids, attention_mask=attention_mask)[0]
    #     Q = self.linear(Q)
    #
    #     mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
    #     Q = Q * mask
    #
    #     return torch.nn.functional.normalize(Q, p=2, dim=2)

    def encode_info(self, input_ids, attention_mask, keep_dims=True):
        assert keep_dims in [True, False, 'return_mask']

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        # print(input_ids.size(), attention_mask.size())
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        # D = self.linear(D)
        # mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
        # D = D * mask

        # D = torch.nn.functional.normalize(D, p=2, dim=2)
        # if self.colbert_config.device == "cuda":
        #     D = D.half()

        # if keep_dims is False:
        #     D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
        #     D = [d[mask[idx]] for idx, d in enumerate(D)]
        #
        # elif keep_dims == 'return_mask':
        #     return D, mask.bool()

        return D

    def score(self, info, config=Config(), cls=False):
        if config.device == "cuda":
            info = info.cuda()
        if cls:
            repr = info[:, 0, :]
        else:
            repr = torch.sum(info, dim=1)
        scores = self.linear_to_1(repr)
        return scores

    def mask(self, input_ids, skiplist):
        mask = [[(x not in skiplist) and (x != self.pad_token) for x in d] for d in input_ids.cpu().tolist()]
        return mask