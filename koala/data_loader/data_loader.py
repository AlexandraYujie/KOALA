import os
import logging
from torch.utils.data import DataLoader, Dataset
import json
import torch
import random
from typing import Dict, List, Set
import csv
import jsonlines
import ujson
from collections import OrderedDict
from itertools import zip_longest, chain
from koala.model.tokenization import QueryTokenizer, DocTokenizer, T5_Tokenizer
# from data_processing import utils
from koala.data_loader.utils import Loader
# from llm_generator.data_utils import filter_training_idxes_func


class MixedSampleDataset(Dataset):
    def __init__(self, config, action, filter_training_idxes=True):  # set a dataset
        assert action in {"training", "validation","test"}
        self.config = config
        self.action = action
        self.query = Loader.load_queries(self.config.queries_root)
        self.collection, self.samples, self.domains = self.get_collection_samples(filter_training_idxes)

    def get_root_of_collection_samples(self):
        if self.action == "training":
            collection_roots = self.config.collection_train_roots_domain_mix
            triples_roots = self.config.triples_train_roots_domain_mix
        if self.action == "validation":
            collection_roots = self.config.collection_valid_roots_domain_mix
            triples_roots = self.config.triples_valid_roots_domain_mix
        if self.action == "test":
            collection_roots = self.config.collection_test_roots_domain_mix
            triples_roots = self.config.triples_test_roots_domain_mix
        return collection_roots, triples_roots

    def get_collection_samples(self, filter_training_idxes):
        collection_roots, triples_roots = self.get_root_of_collection_samples()
        total_collection = []
        total_samples = []
        total_domains = []
        trg_domains = self.config.present_domains
        assert len(collection_roots) == len(triples_roots)
        for i in range(len(collection_roots)):
            collection_root_of_domain = collection_roots[i]
            triples_root_of_domain = triples_roots[i]
            domain = trg_domains[i]
            print("load from:", collection_root_of_domain, "\n", triples_root_of_domain)
            collection_of_domain, samples_of_domain = self.load_data(collection_root_of_domain, triples_root_of_domain)
            # if filter_training_idxes and self.action in {"test"}:  # only test
            #     assert domain in triples_root_of_domain and domain in collection_root_of_domain
            #     samples_of_domain, removed_pair_idx_list = filter_training_idxes_func(domain, self.config.indicator_name, collection_of_domain, samples_of_domain)
            collection_len = len(total_collection)  # 0
            samples_of_domain = [[idx + collection_len for idx in sample] for sample in samples_of_domain]
            total_collection.extend(collection_of_domain)
            total_samples.extend(samples_of_domain)
            total_domains.extend([domain]*len(samples_of_domain))
        assert len(total_domains) == len(total_samples)
        return total_collection, total_samples, total_domains

    def load_data(self, collection_root, triples_root):
        collection = Loader.load_collection(collection_root)
        samples = Loader.load_samples(triples_root)
        return collection, samples


    def process(self, sample):
        passages = [self.collection[pid] for pid in sample]
        return passages

    def __getitem__(self, index):  # dataloader use the func to get data
        return self.process(self.samples[index]), self.domains[index]

    def __len__(self):
        return len(self.samples)  # return batch length


class SampleDataset(Dataset):
    def __init__(self, config, action, filter_training_idxes=True, idx_conversion_mode=True):  # set a dataset
        assert action in {"training", "validation","test"}
        self.config = config
        # if not self.config.few_sample_train:  # few sample mode 已经提前生成好了
        #     if not all([os.path.exists(path) for path in [self.config.triples_train_root,
        #                                                   self.config.collection_train_root,
        #                                                   ]]):
        #         self.collect_data()
        self.collection, self.query, self.samples = self.load_data(action=action)  # load the data
        # if filter_training_idxes and action in {"test"}:  # only test
        #     if filter_training_idxes:
        #         self.samples, removed_pair_idx_list = filter_training_idxes_func(self.config.domain,
        #                                                                          self.config.indicator_name,
        #                                                                          self.collection, self.samples)



    # def collect_data(self):
    #     utils.collect_data_main(raw_data_root_list=self.config.raw_data_root_list, overturn_select=True,
    #                             config=self.config)
    def load_data(self, action):
        query = Loader.load_queries(self.config.queries_root)
        if action == "training":
            collection = Loader.load_collection(self.config.collection_train_root)
            samples = Loader.load_samples(self.config.triples_train_root)
        elif action == "validation":
            collection = Loader.load_collection(self.config.collection_valid_root)
            samples = Loader.load_samples(self.config.triples_valid_root)
        elif action == "test":
            collection = Loader.load_collection(self.config.collection_test_root)
            samples = Loader.load_samples(self.config.triples_test_root)
        return collection, query, samples


    def process(self, sample):
        passages = [self.collection[pid] for pid in sample]
        return passages

    def __getitem__(self, index):  # dataloader use the func to get data
        return self.process(self.samples[index])

    def __len__(self):
        return len(self.samples)  # return batch length


class InferenceDataset(Dataset):
    def __init__(self, config, action, collection_inference_root=None, collection=None, idxes=None):  # set a dataset
        assert action == "inference"
        self.config = config
        if collection != None:
            assert idxes != None
            self.collection, self.idxes = collection, idxes
            self.query = self.load_data(only_query=True)
        else:
            if collection_inference_root != None:
                self.collection_inference_root = collection_inference_root
            else:
                self.collection_inference_root = self.config.collection_inference_root
            self.collection, self.idxes, self.query = self.load_data()  # load the data

    def load_data(self, only_query=False):
        query = Loader.load_queries(self.config.queries_root)
        if only_query:
            return query
        collection, idxes = Loader.load_inference_collection(self.collection_inference_root)  # data_dir/collection.tsv
        return collection, idxes, query

    def __getitem__(self, index):  # dataloader use the func to get data
        return self.collection[index]  # [idx, text]

    def __len__(self):
        return len(self.collection)  # return batch length

class DistillDataset(Dataset):
    def __init__(self, config, action, texts, trg_values):
        assert action in {"training", "validation"}
        assert len(texts) == len(trg_values)
        self.action = action
        self.config = config
        self.query = Loader.load_queries(self.config.queries_root)
        self.texts, self.trg_values = texts, trg_values

    def get_train_valid(self, texts, trg_values, split_ratio=0.8):  # 不用
        data_len = len(texts)
        split_pos = int(split_ratio * data_len)
        if self.action == "training":
            texts = texts[:split_pos]
            trg_values = trg_values[:split_pos]
        elif self.action == "validation":
            texts = texts[split_pos:]
            trg_values = trg_values[split_pos:]
        return texts, trg_values

    def __getitem__(self, index):
        return self.texts[index], self.trg_values[index]

    def __len__(self):
        return len(self.texts)


class Dataloader(DataLoader):
    def __init__(self, action, dataset, batch_size, method, shuffle=None, query_tokenizer=None, doc_tokenizer=None, no_query=False, append_value=False, pair_mode=False):  # 先初始化好的东西
        self.action = action
        self.dataset = dataset
        print(len(dataset),"%%%"*20)
        print()
        self.query_tokenizer = query_tokenizer
        self.doc_tokenizer = doc_tokenizer
        self.method = method
        self.pair_mode = pair_mode  # 1 and 0 0
        self.append_value = append_value
        if no_query:
            self.query = None
        else:
            self.query = dataset.query
        if shuffle == None:  # shuffle not in {True, False}
            shuffle = self.shuffle
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=0, collate_fn=self.collate, drop_last=False)

    def process_queries_passages(self, passages):
        sample_num = len(passages)  # 4
        if self.pair_mode:
            passages = list(chain(*passages))  # 8, 将passages拉成一长条
        if self.query != None:
            queries = [self.query] * sample_num  # 4, 因为queries是四个一样的所以乘以四，假设同一个pair拥有相同的query
        else:
            queries = None
        if self.method in {"ColBERT"}:
            D_ids, D_masks = self.doc_tokenizer.tensorize(passages)  # torch.Size([1, 122]); torch.Size([8, 220])
            Q_ids, Q_masks = self.query_tokenizer.tensorize(queries)
            return (Q_ids, Q_masks), (D_ids, D_masks)
        elif self.method in {"BERT"}:
            if self.query != None:
                if self.pair_mode:
                    queries = [sentence for sentence in queries for _ in range(2)]
                input = [i + j for (i,j) in list(zip(queries, passages))]
                ids, masks = self.doc_tokenizer.tensorize(input)
            else:
                ids, masks = self.doc_tokenizer.tensorize(passages)
            return (ids, masks)  # sample_mode: 4 query and 8 passage ; else: 4 query and 4 passage
        elif self.method in {"T5"}:
            if self.query != None:
                if self.pair_mode:
                    queries = [sentence for sentence in queries for _ in range(2)]
            # input_ids = tokenizer(, return_tensors="pt").input_ids.to("cuda")
            ids, masks, labels = self.doc_tokenizer.tensorize(queries, passages)
            return ids, masks, labels
        elif self.method in {"sbert"}:  # queries是4条，passages是4条
            return queries, passages

    def convert_batch(self,batch):
        # [((x,y),z),((x,y),z),((x,y),z)]
        # convert to ([x,y],[x,y],[x,y]),(z,z,z)
        passages = [sample for (sample, z) in batch]
        z_values = tuple(z for (sample, z) in batch)
        return passages, z_values

    def collate(self, batch):
        if self.append_value:  # distill or mix_data
            passages, values = self.convert_batch(batch)
            processed_input = self.process_queries_passages(passages)
            return processed_input, values
        else:
            passages = batch
            processed_input = self.process_queries_passages(passages)
            return processed_input


    @classmethod
    def get_data_loader(cls, config, action, shuffle=None, **args):  # text trg_values
        query_tokenizer, doc_tokenizer = None, None
        if config.method not in {"sbert", "T5"}:
            query_tokenizer = QueryTokenizer(config)
            doc_tokenizer = DocTokenizer(config)
        elif config.method in {"T5"}:
            doc_tokenizer = T5_Tokenizer(config)
            query_tokenizer = None

        if action == "inference":
            collection_inference_root = args.get("collection_inference_root", None)
            collection = args.get("collection", None)
            idxes = args.get("idxes", None)
            dataset = InferenceDataset(config, action, collection_inference_root=collection_inference_root,
                                       collection=collection, idxes=idxes)
            append_value, pair_mode = False, False
        elif config.domain_mix or (config.distill and action == "test") or (config.few_sample_training and action == "test"):
            if action in {"training", "validation", "test"}:
                dataset = MixedSampleDataset(config, action)  # 定义dataset
                append_value, pair_mode = True, True
        elif config.distill:
            if action in {"training", "validation"}:
                dataset = DistillDataset(config, action, texts=args["texts"], trg_values=args["trg_values"])
                append_value, pair_mode = True, False
        else:
            if action in {"training", "validation", "test"}:
                dataset = SampleDataset(config, action)  # 定义dataset
                append_value, pair_mode = False, True

        batch_size = config.batch_size
        method = config.method
        no_query = config.no_query

        return cls(action, dataset, batch_size, method, shuffle, query_tokenizer, doc_tokenizer, no_query, append_value, pair_mode)

    @property
    def shuffle(self):
        if self.action == "training":
            return True
        elif self.action in {"validation", "inference", "test"}:
            return False
        else:
            raise RuntimeError("error with shuffle setting.")