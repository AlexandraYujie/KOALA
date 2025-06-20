import json
import os.path
import itertools
from typing import Dict
import pickle
import re
import csv
import ast

convert_dic = lambda dic: {v: k for k, v in dic.items()}


def load_text(data_path):
    with open(data_path, "r", encoding="utf-8") as fw:
        text = fw.read()
    return text.strip()


def load_json(data_path, convert=False):
    with open(data_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if convert:
        assert isinstance(data, Dict)
        data = convert_dic(data)
    return data


def save_jsonl(dic_list, file_path):
    with open(file_path, "w", encoding="utf-8") as fw:
        for event in dic_list:
            json.dump(event, fw, ensure_ascii=False)
            fw.write('\n')
    print(f">>> save jsonl to {file_path}: {len(dic_list)}")


def save_json(dic, file_path):
    with open(file_path, "w", encoding="utf-8") as fw:
        json.dump(dic, fw, ensure_ascii=False)
    print(f">>> save json to {file_path}: {len(dic)}")


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f">>> make dir: {dir}")

def save_tsv(data, data_path):
    with open(data_path, 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(data)

def load_tsv(tsv_path):
    collection = []
    with open(tsv_path, encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx % (1000) == 0:
                print(f'{line_idx // 1000}K', end=' ', flush=True)
            pid, passage = line.strip('\n\r ').split('\t')
            assert pid == 'id' or int(pid) == line_idx, f"pid={pid}, line_idx={line_idx}"
            collection.append(passage)
    return collection


def load_jsonl(data_path):
    data = []
    with open(data_path, 'r', encoding="utf-8") as f:
        for line in f:
            try:
                l_json = json.loads(line)
                data.append(l_json)
            except:
                print("[Error loading jsonl] Error in load_jsonl:", line, data_path, sep="\n")
                continue
    return data


def expand_triplets(triplet):
    converted = []
    for elem in triplet:
        if isinstance(elem, list):
            converted.append(elem)
        else:
            converted.append([elem])
    product = itertools.product(*converted)
    return [list(p) for p in product]


def get_value2idx(str_list):
    return {name: idx for idx, name in enumerate(str_list)}


def save_idxes(int_list, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(int_list, f)


def load_idxes(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def extract_dic(text):
    '''
        return None or {} or {empty}
    '''
    text = text.replace("null", "None")
    text = text.replace(": true", ": True")
    text = text.replace(": false", ": False")
    text = text.replace(":true", ": True")
    text = text.replace(":false", ": False")
    match = re.search(r"\{(.*)\}", text, re.DOTALL)  # r'\{.*?\}'
    if match:
        dict_str = match.group()
        try:
            parsed_dict = ast.literal_eval(dict_str)
            # parsed_dict = json.loads(dict_str)
            return parsed_dict  # Output: {'name': 'Alice', 'age': 30}
        except Exception as e:
            print("Error occured in dic extraction:", e)
            print(dict_str)
            return None
    else:
        return None

def extract_list(text):
    '''
        return None or [] or [elements]
    '''
    text = text.replace("null", "None")
    match = re.search(r"\[(.*)\]", text, re.DOTALL)
    if match:
        list_str = match.group()
        try:
            parsed_list = ast.literal_eval(list_str)
            if isinstance(parsed_list, list):
                return parsed_list
            else:
                print("Extracted content is not a list.")
                return None
        except Exception as e:
            print("Error occurred in list extraction:", e)
            print(list_str)
            return None
    else:
        return None


def post_process_decorator(post_process_func):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            resp, prompt = func(self, *args, **kwargs)
            return post_process_func(self, resp, prompt)

        return wrapper

    return decorator


class ParsedResponseError(Exception):
    def __init__(self, result):
        self.result = result

def extract_dics(text):
    '''
    return [empty] or [{}, ]
    '''
    # 改进的正则表达式
    text = text.replace("null", "None")
    pattern = r'\{[^}]*\}'
    matches = re.findall(pattern, text)

    # 使用 ast.literal_eval 将字符串转换为字典
    events = []
    for match in matches:
        try:
            parsed_dict = ast.literal_eval(match)
            events.append(parsed_dict)
        except Exception as e:
            print("Error occured in dic extraction:", e)
            print(match)
            continue

    return events