import json
import os
import logging
import re
import random
import torch
import numpy as np
import yaml
from typing import Dict
import subprocess
from typing import List

level_priority = {"中央":1, "省级":2, "市级":3}

def flatten_list_of_list(list_of_list):
    flattened_list = []
    for l in list_of_list:
        flattened_list += l
    return flattened_list


def mkdir(root):
    if not os.path.exists(root):
        os.makedirs(root)

def get_batch_paras(text_to_be_ranked):
    text_to_be_ranked_split = re.split(r"<\d>：", text_to_be_ranked)
    batch_paras = []
    for split_part in text_to_be_ranked_split:
        split_part = split_part.strip()
        if split_part:
            batch_paras.append(split_part)
    batch_paras = batch_paras[1:]
    if len(batch_paras) == 2:
        return batch_paras

def rank_level(levels):
    try:
        levels = [(level, level_priority[level]) for level in levels]
    except:
        levels = [(level, level_priority[level]) for level in levels.val]
    levels = sorted(levels, key=lambda x:x[1], reverse=False)  # from small to big
    levels = [i[0] for i in levels]
    return levels

def set_seed(seed):
    print("set seed...")
    random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的   　　
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子；
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def read_yml(root):
    assert root.endswith(".yml"), "The config should be yml format."
    with open(root, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    return result

def amend_config_by_args(config, args, filter_keys=None):
    args_dic = args.__dict__
    args_dic = dict(filter(lambda x: x[1] != None, args_dic.items()))  # 剔除未设置的参数
    if filter_keys != None:
        if not isinstance(filter_keys, list):
            raise TypeError("filter_keys is List")
        args_dic = dict(filter(lambda x: x[0] not in filter_keys, args_dic.items()))
    amend_config_by_dic(config, args_dic)

def amend_config_by_dic(config, config_dic: Dict):
    keys = config_dic.keys()
    unrecognized_set = config.configure(**config_dic, ignore_unrecognized=True)
    print(">>> unrecognized set para in args:", unrecognized_set)
    set_key = keys - unrecognized_set
    print(">>> set key:", {k:v for k,v in config_dic.items() if k in set_key})

def check_config(config):
    if config.no_query:
        assert not config.method in {"ColBERT"}, "no no_query mode in this method"
    assert (config.distill + config.domain_mix + config.few_sample_training) in {1, 0}  # [1, 1] == 1
    # assert config.base_model_root in config.checkpoint, "all of the base models in models/ root"

def load_dic(root):
    print(root)
    with open(root, "r", encoding="utf-8") as fp:
        try:
            dic = json.load(fp, strict=False)
        except:
            dic = json.loads(fp.read(), strict=False)
    return dic

def write_config():  # TODO: write more predefined config setting.
    config = {
        "method" : "BERT",
        "checkpoint" : "bert-base-chinese",
        "batch_size" : 2,
        "epoch" : 3,
    }
    with open('BERT_config.yml', 'w', encoding='utf-8') as f:
        yaml.dump(data=config, stream=f, allow_unicode=True)

def launch_cmd(cmd, working_dir):
    p = subprocess.Popen(cmd, cwd=working_dir, shell=True)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError()


def save_data_list(saving_root: str, data_list: List[str]) -> None:
    with open(saving_root, "w", encoding="utf-8") as fw:
        print(f">>>>> length of data: {len(data_list)}")
        for data in data_list:
            if isinstance(data, dict):
                json.dump(data, fw, ensure_ascii=False)
            else:
                fw.write(data)
            fw.write("\n")
    print(f">>> saved data_list to {saving_root}")

def load_data_list(data_root:str) -> List[str]:
    data_list = []
    with open(data_root, "r", encoding="utf-8") as fp:
        while True:
            line = fp.readline().strip()
            if not line:
                break
            data_list.append(line)
    print(f">>>>> load length of data: {len(data_list)}")
    return data_list

if __name__ == "__main__":
    write_config()