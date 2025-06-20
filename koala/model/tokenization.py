import torch
from koala.model.utils import get_max_len_of_queries
from koala.model.base_bert import class_factory
from transformers import AutoTokenizer
import numpy as np

class QueryTokenizer():
    def __init__(self, config):
        base_BERT = class_factory(config.checkpoint)  # 先找到类，然后继承这个类，加载的时候里面的config和tokenizer全部都是用Auto类
        self.tok = base_BERT.raw_tokenizer_from_pretrained(config.checkpoint)
        self.config = config
        self.query_maxlen = config.query_maxlen

        self.Q_marker_token, self.Q_marker_token_id = config.query_token, self.tok.convert_tokens_to_ids(
            config.query_token_id)
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id
        self.pad_token, self.pad_token_id = self.tok.pad_token, self.tok.pad_token_id
        self.used = False

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]  # 分词

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix + [self.mask_token] * (self.query_maxlen - (len(lst) + 3)) for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        ids = self.tok(batch_text, add_special_tokens=False)['input_ids']

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix + [self.mask_token_id] * (self.query_maxlen - (len(lst) + 3)) for lst in ids]

        return ids

    def decode(self, batch_ids):
        assert type(batch_ids) in [list, tuple], (type(batch_ids))

        tokens = [self.tok.decode(x) for x in batch_ids]  # 分词
        return tokens

    def tensorize(self, batch_text):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        full_length_search = self.config.full_length_search
        # add placehold for the [Q] marker
        batch_text = ['. ' + x for x in batch_text]

        # Full length search is only available for single inference (for now)
        # Batched full length search requires far deeper changes to the code base
        # assert (full_length_search == False or (type(batch_text) == list and len(batch_text) == 1))

        if full_length_search:  # 如果只有一条的话（一个batch中只有一条query的话）
            max_length = get_max_len_of_queries(self.tok, batch_text, self.query_maxlen) + 2
        else:
            # Max length is the default max length from the config
            max_length = self.query_maxlen

        obj = self.tok(batch_text, padding='max_length', truncation=True,
                       return_tensors='pt', max_length=max_length)  # max_length，大于它的截断，小于它的填充

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == self.pad_token_id] = self.mask_token_id



        if self.config.attend_to_mask_tokens:
            mask[ids == self.mask_token_id] = 1
            assert mask.sum().item() == mask.size(0) * mask.size(1), mask

        if self.used is False:
            self.used = True

            print()
            print("#> QueryTokenizer.tensorize(batch_text[0], batch_background[0], bsize) ==")
            print(f"#> Input: {batch_text[0]}")
            print(f"#> Output IDs: {ids[0].size()}, {ids[0]}")
            print(f"#> Output Mask: {mask[0].size()}, {mask[0]}")
            print(f"#> Decode: {self.decode(list(ids.detach().numpy()[0]))}")
            print()

        return ids, mask



class DocTokenizer():
    def __init__(self, config):
        Base_BERT = class_factory(config.checkpoint)
        self.tok = Base_BERT.raw_tokenizer_from_pretrained(config.checkpoint)

        self.config = config
        self.doc_maxlen = config.doc_maxlen

        self.D_marker_token, self.D_marker_token_id = self.config.doc_token, self.tok.convert_tokens_to_ids(self.config.doc_token_id)
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.D_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix for lst in tokens]

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        ids = self.tok(batch_text, add_special_tokens=False)['input_ids']

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.D_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix for lst in ids]

        return ids

    def tensorize(self, batch_text):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [D] marker
        batch_text = ['. ' + x for x in batch_text]

        obj = self.tok(batch_text, padding='longest', truncation='longest_first',
                       return_tensors='pt', max_length=self.doc_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [D] marker
        ids[:, 1] = self.D_marker_token_id

        return ids, mask

class T5_Tokenizer():
    def __init__(self, config):
        print(config.checkpoint)
        self.tok = AutoTokenizer.from_pretrained(config.checkpoint)
        self.doc_maxlen = config.doc_maxlen
        self.query_maxlen = config.query_maxlen
        self.extra_tok_id = self.tok("<extra_id_10>").input_ids[0]

    def _get_input_texts(self, queries, passages):
        if queries != None:
            assert len(queries) == len(passages)
        # passage_max_length = get_max_len_of_queries(self.tok, passages, self.query_maxlen)
        input_texts = []
        for i in range(len(passages)):
            passage = passages[i]
            if queries != None:
                query = queries[i]
                input_text = f"Standard:{query}. Document:{passage[:self.doc_maxlen]}. Relevant:"
            else:
                input_text = f"Document:{passage[:self.doc_maxlen]}. Relevant:"
            input_texts.append(input_text)
        return input_texts


    def tensorize(self, queries, passages):  # TODO: substitute for measure_obj
        input_texts = self._get_input_texts(queries, passages)
        obj = self.tok(input_texts, padding='longest', truncation='longest_first',
                       return_tensors='pt')
        ids, masks = obj['input_ids'], obj['attention_mask']
        labels = torch.tensor([[id] for id in [self.extra_tok_id]*len(passages)])
        # labels = self.tok(["true","false"]*(len(queries)/2), return_tensors="pt", padding='max_length', truncation=True).input_ids

        # self.tok.convert_tokens_to_ids('<extra_id_0>')
        return ids, masks, labels