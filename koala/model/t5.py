import torch
from transformers import MT5ForConditionalGeneration, AutoTokenizer, MT5EncoderModel
import torch.nn.functional as F
import torch.nn as nn
import os
import json
def get_base_class(name):
    name_class_dic = {
        "mt5-base": MT5ForConditionalGeneration
    }
    base_class = name_class_dic.get(name, None)
    if base_class != None:
        return base_class
    else:
        raise RuntimeError("do not find the basic class of T5")
class T5Model(torch.nn.Module):
    def __init__(self, config,**args):
        super(T5Model, self).__init__()
        self.config = config
        self.only_encoder = self.config.only_encoder
        self.hidden_size = self._get_hidden_size()
        if self.only_encoder:
            self.model = MT5EncoderModel.from_pretrained(self.config.checkpoint)
            self.linear_to_1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.config.dim),  # TODO: I do not know it is true or false
            nn.ReLU(),
            nn.Linear(self.config.dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        else:
            base_class = get_base_class(self.config.base_model_name)
            self.model = base_class.from_pretrained(self.config.checkpoint)
        self.tok = AutoTokenizer.from_pretrained(self.config.checkpoint)
        self.true_tok = self.tok("true").input_ids[0]
        self.false_tok = self.tok("false").input_ids[0]
        self.extra_tok = self.tok("<extra_id_10>").input_ids[0]
        self.sigmoid_score = config.sigmoid_score

    def _get_hidden_size(self):
        transformer_json_path = os.path.join(self.config.checkpoint, 'config.json')
        if not os.path.exists(transformer_json_path):
            raise RuntimeError(f"No {transformer_json_path} file.")
        with open(transformer_json_path) as fp:
            transformer_config = json.load(fp)
            hidden_size = transformer_config.get("d_model", None)
            if hidden_size == None:
                raise RuntimeError(f"No d_model parameter in {transformer_json_path}.")
            return hidden_size

    def forward(self, input_ids, attention_mask, labels):
        # input_ids = input_ids.to(self.config.device)
        # outputs = self.model.generate(input_ids)
        # print("*******")
        # print(outputs.size())
        # print(self.tok.decode(outputs[0],skip_special_tokens=True, clean_up_tokenization_spaces=False))
        input_ids = input_ids.to(torch.device(self.config.device))
        labels = labels.to(torch.device(self.config.device))
        attention_mask = attention_mask.to(torch.device(self.config.device))
        if self.only_encoder:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            outputs = torch.sum(outputs, dim=1)
            # outputs = outputs[:,0,:]
            scores = self.linear_to_1(outputs)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            scores = self.scores(logits)
        if self.sigmoid_score:
            scores = torch.sigmoid(scores)
        return scores
        # return logits
        # loss_fct = CrossEntropyLoss(ignore_index=-100)
        # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

    def scores(self, logits, mode="3"):
        # print("logits size:", logits.size())  # batch_size, 1, vocab_size
        logits = torch.squeeze(logits, dim=1)  # batch_size, vocab_size
        if mode == "1":  # all softmax，get the true tok
            softmax_logits = F.softmax(logits, dim=-1)   # bs, vs
            # print("softmax_size:", softmax_logits.size())
            scores = softmax_logits[:, self.true_tok]  # bs, 1
            # softmax on all, and only true represent the score
        elif mode == "2":  # get the true and false logit, then softmax, make the true as the score
            tf_logits = logits[:, [self.true_tok, self.false_tok]]  # bs, 2
            softmax_tf = F.softmax(tf_logits, dim=-1)  #
            scores = softmax_tf[:, 0]  # bs, 0, true dimension
        elif mode == "3":  # the method of rank t5
            scores = logits[:, self.extra_tok]
        return scores

