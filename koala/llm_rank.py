from .utils.call_llm import ChatClient
from .prompt import *
from functools import partial
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
from collections import defaultdict
import random
from .config.config import RankConfig
from koala.utils.utils import load_jsonl, save_jsonl, save_tsv, mkdir, load_tsv
random.seed(42)
from koala.utils.call_llm import call_chatbot
import warnings
from tqdm import tqdm

def recover_item(collection, grouped_item):
    recovered_item =  [collection[int(i)] for i in grouped_item]
    return recovered_item
    pass

# recover the data
def recover_data(collection, grouped_data):
    recovered_data = []
    for grouped_item in grouped_data:
        recovered_item = recover_item(collection, grouped_item)
        recovered_data.append(recovered_item)
    return recovered_data



class RankByLLM(ChatClient):
    def __init__(self, chatbot_name, api_key, base_url, language, style_name, style_definition, print_resp=False, **kwargs):
        super().__init__(chatbot_name, api_key, base_url, print_resp, **kwargs)
        # self.chatbot = chatbot
        self.language = language
        self.style_name = style_name
        self.style_definition = style_definition
        self.rank_template, self.extract_template = self.init_template()

    def init_template(self):
        if self.language == "cn":
            template = cn_template
            extract_template = f"{self.style_name}更强的文本："
        if self.language == "en":
            template = en_template
            extract_template = f"Text with stronger {style_name}: "
        template_partial = partial(template, self.style_name, self.style_definition)
        return template_partial, extract_template

    def parse_func(self, text):
        pattern = re.escape(self.extract_template) + r"\s*\[?([A-Z])\]?"
        match = re.search(pattern, text)
        if match:
            result = match.group(1).strip()
            if result in {"A", "B"}:
                return result
        return None

    def rank(self, text1, text2):
        '''
        :param text1:
        :param text2:
        :return: it will return "A" or "B" or None
        '''
        rank_template = self.rank_template(text1, text2)
        try:
            rank_output = self.get_valid_structured_output(rank_template,
                                                           parse_func=self.parse_func)  # 能正确根据parse_func解析出结果
        except Exception as e:
            print("[Generate rank_output error!]", e)
            rank_output = None
        return rank_output

    def _compare_pair(self, text1, text2):
        result1 = self.rank(text1, text2)
        result2 = self.rank(text2, text1)
        if result1 == "A" and result2 == "B":
            return 0
        elif result1 == "B" and result2 == "A":
            return 1
        else:
            return None

    def rank_items(self, recovered_item_list, max_workers=8):
        '''
        :param recovered_item_list: [[text1, text2], ...]
        :return: [0, 1, ...]  # 0 refers to the unchanged order, index 0 is bigger.
        '''
        rank_output_list = [None] * len(recovered_item_list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._compare_pair, item[0], item[1]): idx
                for idx, item in enumerate(recovered_item_list)
            }

            # 使用 tqdm 包装 as_completed，展示进度
            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Ranking"):
                idx = future_to_index[future]
                try:
                    rank_output_list[idx] = future.result()
                except Exception as e:
                    print(f"Error in future {idx}: {e}")
                    rank_output_list[idx] = None
        return rank_output_list



def load_data(config):
    domain2material = {}
    for domain in os.listdir(config.processed_data_dir):
        processed_data_domain_dir = config.processed_data_dir + f"{domain}/"
        # print(processed_data_domain_dir)

        collection_path = processed_data_domain_dir + config.collection_file_name
        grouped_data_path = processed_data_domain_dir + config.grouped_data_file_name

        if not os.path.exists(grouped_data_path) or not os.path.exists(collection_path):
            warnings.warn(message=f"The dir not ready in {processed_data_domain_dir}")
            continue

        collection = load_tsv(collection_path)
        print(f"loaded collection for domain {domain}", len(collection))

        grouped_data = load_jsonl(grouped_data_path)
        print(f"loaded grouped_data for domain {domain}", len(grouped_data))
        # recover the data
        # recovered_data = recover_data(collection, grouped_data)
        print("\n")

        domain2material[domain] = {"collection": collection, "grouped_data": grouped_data}
    return domain2material



def split_data(config):
    for root, dirs, files in os.walk(config.processed_data_dir):
        for file in files:
            if config.ranked_grouped_data_file_name in file:
                grouped_data_ranked_path = root + "/" + file
                print("start to split:", grouped_data_ranked_path)
                grouped_data = load_jsonl(grouped_data_ranked_path)
                # split it!
                training_num = int(len(grouped_data) * config.train_ratio)
                validation_num = len(grouped_data) - training_num
                print(training_num, validation_num)
                grouped_data_training = grouped_data[:training_num]
                grouped_data_validation = grouped_data[training_num:]
                train_path = root + "/" + config.ranked_grouped_train_data_file_name
                valid_path = root + "/" + config.ranked_grouped_valid_data_file_name
                save_jsonl(grouped_data_training, train_path)
                save_jsonl(grouped_data_validation, valid_path)
                print("\n")

def rank(config: RankConfig):
    # chatbot = call_chatbot(llm_model=config.llm_model, base_url=config.base_url, api_key=config.api_key)
    # chatbot = ChatClient(chatbot_name="qwen-plus", print_resp=True, rate_limit_count=1155, rate_limit_period=60)
    rank_machine = RankByLLM(chatbot_name="qwen-plus",
                             print_resp=config.print_resp,
                             language=config.language,
                             style_name=config.style_name,
                             style_definition=config.style_definition,
                             api_key=config.api_key,
                             base_url=config.base_url)

    domain2material = load_data(config)


    for domain, material in domain2material.items():
        if domain not in config.trg_domains:
            continue

        print(f"Start to process domain: {domain}...")
        processed_data_domain_dir = config.processed_data_dir + f"{domain}/"
        collection = material["collection"]
        grouped_data = material["grouped_data"]
        recovered_item_list = []
        for idx, grouped_item in enumerate(grouped_data):
            if config.data_num and idx == config.data_num:
                break
            recovered_item = recover_item(collection, grouped_item)
            recovered_item_list.append(recovered_item)

        rank_output_list = rank_machine.rank_items(recovered_item_list=recovered_item_list, max_workers=config.max_workers)
        assert len(rank_output_list) == len(recovered_item_list)

        ranked_grouped_data = []
        for idx, rank_output in enumerate(rank_output_list):
            assert rank_output in {0, 1, None}
            grouped_item = grouped_data[idx]
            if rank_output == 0:
                ranked_grouped_data.append(grouped_item)
            elif rank_output == 1:
                ranked_grouped_data.append(grouped_item[::-1])
            else:
                continue
        save_jsonl(ranked_grouped_data, processed_data_domain_dir + config.ranked_grouped_data_file_name)

    split_data(config)