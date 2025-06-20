import os
import json
from collections import defaultdict
import random
from .utils.utils import load_jsonl, save_jsonl, save_tsv, mkdir, load_tsv
from .config.config import Config

random.seed(42)


from itertools import combinations

def group_pairwise_combinations(texts, n):
    """
    Group the `texts` list into subgroups of `n` elements each, and generate all pairwise combinations within each group.

    Parameters:
        texts (list): The input list of texts.
        n (int): The number of elements in each group.

    Returns:
        list: A list containing all pairwise combinations from each group, where each combination is in the form [[a, b], [a, c], ...].
"""
    result = []
    for i in range(0, len(texts), n):
        group = texts[i:i + n]
        if len(group) < 2:
            continue
        pairs = [list(pair) for pair in combinations(group, 2)]
        result.extend(pairs)
    return result


def prepare_unrank_data(config: Config):
    raw_data = load_jsonl(config.raw_data_path)
    domain2texts = defaultdict(list)
    for data in raw_data:
        domain2texts[data["domain"]].append(data["text"])

    for domain, texts in domain2texts.items():
        idxes = list(range(len(texts)))
        collection = list(zip(idxes, texts))
        pairwise_data = group_pairwise_combinations(idxes, n=4)
        processed_data_domain_dir = config.processed_data_dir + f"{domain}/"
        mkdir(processed_data_domain_dir)
        save_tsv(collection, processed_data_domain_dir + config.collection_file_name)
        save_jsonl(pairwise_data, processed_data_domain_dir + config.grouped_data_file_name)


if __name__ == "__main__":
    prepare_unrank_data()