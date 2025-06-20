from typing import List
from koala.config.core_config import DefaultVal
from dataclasses import dataclass
from torch.cuda import is_available
import logging
from koala.utils.training import rank_level, mkdir
from datetime import datetime
import os


@dataclass
class ExpSetting:
    '''
    :param random_query: currently, it is only effective to method sbert, by modifying the <class: SbertBase>
    '''
    domain_mix: bool = DefaultVal(False)  # 控制 triples & collection
    distill: bool = DefaultVal(False)
    no_query: bool = DefaultVal(False)  # 正常情况下是有query的
    no_test: bool = DefaultVal(False)
    random_query: bool = DefaultVal(False)
    no_model_saving: bool = DefaultVal(False)
    inference: bool = DefaultVal(False)
    sigmoid_score: bool = DefaultVal(False)  # not used in sbert
    no_linear_add: bool = DefaultVal(False)  # only used in sbert, 默认是add linear
    few_sample_training: bool = DefaultVal(False)
    crossentropy_loss: bool = DefaultVal(False)

    @property
    def return_domain_result(self):
        return (self.distill or self.domain_mix or self.few_sample_training)

    @property
    def no_result_saving(self):
        return False

@dataclass
class DataSetting(ExpSetting):
    domain: str = DefaultVal("edu")
    processed_data_dir: str = DefaultVal(f"data/processed/training/")
    processed_test_data_dir: str = DefaultVal(f"data/processed/test/")
    grouped_data_file_name: str = DefaultVal("grouped_data.jsonl")
    ranked_grouped_data_file_name: str = DefaultVal("grouped_data.ranked.jsonl")
    ranked_grouped_train_data_file_name: str = DefaultVal("grouped_data.ranked.train.jsonl")
    ranked_grouped_valid_data_file_name: str = DefaultVal("grouped_data.ranked.valid.jsonl")
    collection_file_name: str = DefaultVal("collection.tsv")
    raw_data_path: str = DefaultVal("data/raw/raw_data.jsonl")
    unified_query_id: int = DefaultVal("0")
    language: str = DefaultVal("cn")
    mixed_domains: List[str] = DefaultVal(["edu", "eco", "liv", "tech", "cul", "med", "emp", "econ", "agri", "gov"])



    # @property  # it won't be init
    # def raw_data_root_list(self):
    #     raw_data_root_list = [f"../../human_evaluation/generated_result/{self.domain}_{self.mode}"]
    #     return raw_data_root_list

    def get_data_dir(self, domain):
        # level_name = "-".join(rank_level(self.levels))
        data_dir = self.processed_data_dir + f"{domain}/"  # _{level_name}_{self.mode}_{self.indicator_name}
        if not os.path.exists(data_dir):
            raise RuntimeError(f"no file {data_dir}")
        return data_dir

    @property
    def data_dir(self):
        data_dir = self.get_data_dir(self.domain)
        return data_dir

    # def get_valid_data_dir(self, domain):
    #     data_dir = f"data/processed/validation/{domain}/"  # _{level_name}_{self.mode}_{self.indicator_name}
    #     if not os.path.exists(data_dir):
    #         raise RuntimeError(f"no file {data_dir}")
    #     return data_dir

    # @property
    # def valid_data_dir(self):
    #     valid_data_dir = self.get_valid_data_dir(self.domain)
    #     return valid_data_dir

    def get_target_roots(self, file_name):
        target_roots = []
        for domain in self.mixed_domains:
            data_dir_of_domain = self.get_data_dir(domain)
            root_of_domain = data_dir_of_domain + file_name
            assert os.path.exists(root_of_domain)
            target_roots.append(root_of_domain)
        return target_roots

    @property
    def triples_train_roots_domain_mix(self):
        return self.get_target_roots(self.ranked_grouped_data_file_name)

    @property
    def triples_valid_roots_domain_mix(self):
        return self.get_target_roots(self.ranked_grouped_data_file_name)

    @property
    def collection_train_roots_domain_mix(self):
        return self.get_target_roots(self.collection_file_name)

    @property
    def collection_valid_roots_domain_mix(self):
        return self.get_target_roots(self.collection_file_name)

    @property
    def present_domains(self):
        # if not self.supplement_domains_test:
        present_domains = self.mixed_domains
        # else:
        #     present_domains = self.supplement_domains
        return present_domains

    @property
    def triples_test_roots_domain_mix(self):
        mixed_domains = self.present_domains

        triples_test_roots_domain_mix = [self.processed_test_data_dir + f"{domain}/{self.ranked_grouped_data_file_name}" for domain in mixed_domains]
        for root in triples_test_roots_domain_mix:
            assert os.path.exists(root)
        return triples_test_roots_domain_mix

    @property
    def collection_test_roots_domain_mix(self):
        mixed_domains = self.present_domains
        collection_test_roots_domain_mix = [self.processed_test_data_dir + f"{domain}/{self.collection_file_name}" for domain in mixed_domains]
        for root in collection_test_roots_domain_mix:
            assert os.path.exists(root)
        return collection_test_roots_domain_mix


    @property
    def triples_train_root(self):
        if not self.few_sample_training:
            triples_train_root = self.data_dir + self.ranked_grouped_train_data_file_name
        else:
            triples_train_root = "data/labeled_data_collected/training_idxes/triples_train.jsonl"
        print(">>> triples_train_root:",triples_train_root)
        return triples_train_root

    @property
    def collection_train_root(self):
        if not self.few_sample_training:
            collection_train_root = self.data_dir + self.collection_file_name
        else:
            collection_train_root = f"data/labeled_data_collected/training_idxes/{self.collection_file_name}"
        print(">>> collection_train_root:", collection_train_root)
        return collection_train_root

    @property
    def collection_valid_root(self):
        if not self.few_sample_training:
            collection_valid_root = self.data_dir + self.collection_file_name
        else:
            collection_valid_root = f"data/labeled_data_collected/training_idxes/{self.collection_file_name}"
        print(">>> collection_valid_root:", collection_valid_root)
        return collection_valid_root

    @property
    def triples_valid_root(self):  # validation要根据domain进行分割
        if not self.few_sample_training:
            triples_valid_root = self.data_dir + self.ranked_grouped_train_data_file_name
        else:
            triples_valid_root = "data/labeled_data_collected/training_idxes/triples_valid.jsonl"
        print(">>> triples_valid_root:", triples_valid_root)
        return triples_valid_root

    @property
    def collection_test_root(self):
        collection_test_root = self.processed_test_data_dir + f"{self.domain}/{self.collection_file_name}"  # f"data/labeled_data_collected/consensus_2/collection_{self.domain}_{self.indicator_name}.tsv"
        return collection_test_root

    @property
    def triples_test_root(self):
        triples_test_root = self.processed_test_data_dir + f"{self.domain}/grouped_data.ranked.jsonl"  # f"data/labeled_data_collected/consensus_2/triples_test_{self.domain}_{self.indicator_name}.jsonl"
        return triples_test_root

    @property
    def queries_root(self):
        return f"data/queries/{self.unified_query_id}.txt"

    @property
    def collection_inference_root(self):
        # collection_inference_root = self.collection_test_root
        collection_inference_root = self.data_dir + self.collection_file_name  # 训练+验证数据
        print(">>>> #### collection_inference_root:", collection_inference_root)
        return collection_inference_root
@dataclass
class ResourceSetting(DataSetting):
    method: str = DefaultVal("ColBERT")  # ColBERT
    base_model_root: str = "models/"
    base_model_name: str = DefaultVal("bert-base-chinese")  # bert-base-chinese
    current_datetime = datetime.now()
    formatted_time: str = current_datetime.strftime("%y-%m-%d_%H-%M-%S")
    only_encoder: bool = DefaultVal(False)
    # similarity: str = DefaultVal("cos")

    @property
    def checkpoint(self):
        checkpoint = self.base_model_root + self.base_model_name
        return checkpoint

    @property
    def experiment_name(self):
        if self.domain_mix:
            data_name = "domain_mix"
        elif self.few_sample_training:
            data_name = "few_sample_training"
        else:
            data_dir = self.data_dir
            data_name = os.path.split(os.path.dirname(data_dir))[1]

        experiment_name = "/".join([data_name, self.method, self.base_model_name]) + f"/{self.formatted_time}/"
        return experiment_name

    @property
    def model_dir(self):
        model_dir = "experiment/model/" + self.experiment_name
        mkdir(model_dir)
        return model_dir


    # @property
    # def model_root(self):
    #     model_root = self.model_dir + "model.pt"
    #     return model_root

    @property
    def result_dir(self):
        # if not self.no_result_saving:  # 不这样写是因为我要存test_sample.xlsx 但我不存exp_record.json
        result_dir = "experiment/result/" + self.experiment_name
        mkdir(result_dir)
        return result_dir
        # else:
        #     return None


@dataclass
class TokenizerSettings:
    query_token_id: str = DefaultVal("[unused0]")
    doc_token_id: str = DefaultVal("[unused1]")
    query_token: str = DefaultVal("[Q]")
    doc_token: str = DefaultVal("[D]")
    query_maxlen: int = DefaultVal(122) # 基本上没用
    doc_maxlen: int = DefaultVal(512)  ###
    attend_to_mask_tokens: bool = DefaultVal(False)
    mask_punctuation: bool = DefaultVal(True)
    full_length_search: bool = DefaultVal(True)  # only for query



@dataclass
class TrainSetting:
    seed: int = DefaultVal(12345)
    batch_size: int = DefaultVal(2)  # the true bs = 4*2
    epoch: int = DefaultVal(3)
    lr: float = DefaultVal(3e-06)
    maxsteps: int = DefaultVal(500_000)
    warmup: int = DefaultVal(None)
    warmup_bert: int = DefaultVal(None)
    use_ib_negatives: bool = DefaultVal(False)
    similarity: str = DefaultVal('cos')  # only adaptive to sbert model, required for revision (1123) reviewer3
    interaction: str = DefaultVal('colbert')
    dim: int = DefaultVal(128)
    log_interval: int = DefaultVal(10)


    @property
    def device(self):
        if is_available():
            device = "cuda"
            logging.info("using gpu to train...")
        else:
            device = "cpu"
            logging.info("using cpu to train...")
        return device


    # @property
    # def collection_inference_root(self):
    #     return f"data/test/collection_{self.domain}.tsv"


    # @property
    # def result_dir(self):
    #     model_dir = self.model_dir
    #     result_dir = model_dir.replace("model", "result")
    #     return result_dir

    # @property
    # def output_root(self):
    #     return self.result_dir + f"idx_value_{self.domain}.jsonl"
@dataclass
class DistillSetting:
    current_datetime = datetime.now()
    formatted_time: str = current_datetime.strftime("%y-%m-%d_%H-%M-%S")
    batch_size: int = DefaultVal(4)
    domain_seq: List = DefaultVal(["edu", "liv", "eco", "tech", "cul"])
    proportion: List = DefaultVal(1)
    shuffle: bool = DefaultVal(False)

    @property
    def model_dir(self):
        model_dir = f"experiment/model/distill/{self.formatted_time}/"
        mkdir(model_dir)
        return model_dir

    @property
    def result_dir(self):
        result_dir = f"experiment/result/distill/{self.formatted_time}/"
        mkdir(result_dir)
        return result_dir