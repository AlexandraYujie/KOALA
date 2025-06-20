import os.path
import time

import torch
from collections import OrderedDict
import json
from koala.data_loader.data_loader import Dataloader, MixedSampleDataset, SampleDataset
from koala.data_loader.utils import Loader
# from llm_generator.utils import generate_pair_idx
from .model.colbert import ColBERT
from koala.model.hidden_bert import HiddenBERT
from koala.model.sbert import Sbert
from koala.model.t5 import T5Model
from koala.utils.training import load_dic
from functools import partial
import pandas as pd
import numpy as np
def get_model(model_name):
    model_name_class_dic = {
        "ColBERT": ColBERT,
        "BERT": HiddenBERT,
        "sbert": Sbert,
        "T5": T5Model
    }
    return model_name_class_dic[model_name]


def evaluation(data_loader, model, sample_saving_dir=None, log_interval=2, return_domain_acc=False):  # test & validation
    time_log_list = []

    model = model.to(torch.device("cuda"))
    model.eval()
    pred_acc_list = []
    loss_list = []
    domain_list = []
    append_value = data_loader.append_value
    if return_domain_acc: assert append_value
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            if append_value:
                batch, domains = batch_data
            else:
                batch = batch_data

            start_t = time.time()

            scores = model(*batch)  # forward data

            end_time = time.time() - start_t
            time_log_list.append(end_time)

            scores = scores.view(-1, 2)  # [4,2]
            loss = torch.log(1 + torch.exp(scores[:, 1] - scores[:, 0])).sum()
            loss_list.append(loss.item())
            pred_acc_list_batch, positive_avg, negative_avg = print_progress(scores)
            pred_acc_list.extend(pred_acc_list_batch)
            if append_value:
                domain_list.extend(list(domains))
            if batch_idx % log_interval == 0:
                print(">> pred acc for this batch (evaluation):", sum(pred_acc_list_batch) / len(pred_acc_list_batch))
                print(batch_idx, loss)

    pred_acc = sum(pred_acc_list) / len(pred_acc_list)  # for every epoch
    avg_loss = sum(loss_list)/len(loss_list)
    print(">> pred acc for this epoch (evaluation):", pred_acc)
    print(">> pred loss for this epoch (evaluation):", avg_loss)

    print("***" * 20)
    print("pairwise data num:", len(pred_acc_list))
    print("using total time:", sum(time_log_list))
    print("using averge time:", sum(time_log_list) / len(pred_acc_list))
    print("***" * 20)

    if append_value:
        domain_acc_dic = get_domain_acc(pred_acc_list, domain_list)

    if sample_saving_dir:
        assert not data_loader.shuffle
        assert data_loader.action in {"validation", "test"}
        assert hasattr(data_loader, "dataset")
        dataset = data_loader.dataset
        assert len(dataset.samples) == len(pred_acc_list)

        sample_saving_root = sample_saving_dir + f"{data_loader.action}_sample.xlsx"

        sample_dic = {"idx":[], "text_to_be_ranked":[], "acc":[]}
        if append_value:
            sample_dic["domain"] = []

        for i, data in enumerate(dataset):
            if append_value:
                sample_text, domain = data
            else:
                sample_text = data
            acc = pred_acc_list[i]
            if acc:  # True
                acc = 1
            else:  # False
                acc = 0
            text_to_be_ranked = "\n\n".join(sample_text)
            sample_dic["idx"].append(i)
            sample_dic["text_to_be_ranked"].append(text_to_be_ranked)
            sample_dic["acc"].append(acc)
            if append_value:
                sample_dic["domain"].append(domain)

        pd.DataFrame(sample_dic).to_excel(sample_saving_root, index=False)
        print(f"saved {sample_saving_root}")

    output = (pred_acc, avg_loss)
    if return_domain_acc:
        output = output + (domain_acc_dic, )
    return output




def get_domain_acc(pred_acc_list, domain_list):
    assert len(pred_acc_list) == len(domain_list)
    domain_acc_list_dic = {}
    for i in range(len(pred_acc_list)):
        domain = domain_list[i]
        pred_acc = pred_acc_list[i]
        if domain not in domain_acc_list_dic:
            domain_acc_list_dic[domain] = []
        domain_acc_list_dic[domain].append(pred_acc)
    domain_acc_dic = {}
    for domain, acc_list in domain_acc_list_dic.items():
        domain_acc_dic[domain] = {"acc":sum(acc_list)/len(acc_list),"len": len(acc_list)}
        print(f"domain:{domain} | acc:{sum(acc_list)/len(acc_list)} | length: {len(acc_list)}")
    return domain_acc_dic

def inference(inference_loader, model, config, show_sample=True, log_interval=10):
    model.eval()
    score_list = []
    model.to(torch.device(config.device))
    print(torch.device(config.device))
    with torch.no_grad():
        for batch_idx, batch in enumerate(inference_loader):
            scores = model(*batch)  # forward data
            # print_progress(scores)
            scores = scores.cpu().detach().numpy()
            if batch_idx % log_interval == 0:
                print(batch_idx, scores)
            score_list.extend(scores)
    # save to json
    score_list = [float(i) for i in score_list]
    return score_list


def print_progress(scores):
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    pred_acc = (scores[:, 0] > scores[:, 1]).cpu().detach().numpy()
    return pred_acc, positive_avg, negative_avg

def load_model(config, model=None, pair_mode=True):
    Model = get_model(config.method)  # get class of model
    if config.method in {"ColBERT", "sbert"}:
        model_class = Model(config, pair_mode)  # init model class
    else:
        model_class = Model(config)
    if model == None:
        model_dir = config.model_dir
        load_checkpoint(model_class, model_dir=model_dir)  # only pretrained paras
        print("loaded model from:", model_dir)
    else:
        load_checkpoint(model_class, model=model)
    return model_class

def load_checkpoint(model_class, model_dir=None, model=None):
    if model_dir and not model:
        checkpoint = load_checkpoint_raw(model_dir)
    else:
        checkpoint = model.state_dict()

    try:
        model_class.load_state_dict(checkpoint)
    except:
        print("[WARNING] Loading checkpoint with strict=False")
        model_class.load_state_dict(checkpoint, strict=False)  # ['model_state_dict']

def load_checkpoint_raw(model_dir):
    checkpoint = torch.load(model_dir + "model.pt", map_location='cpu')  # load pra
    return checkpoint  # return paras

def configure_config(config, model_dir):
    print("loaded config from:", model_dir)
    with open(model_dir + "artifact.metadata", "r", encoding="utf-8") as fp:
        artifact_config = json.load(fp)
    artifact_config = replace_config(artifact_config, key_value_replace_list=[("similarity", "cosine", "cos")])
    config.configure(**artifact_config)

def replace_config(artifact_config, key_value_replace_list):
    for key_value_replace in key_value_replace_list:
        key = key_value_replace[0]
        value = artifact_config.get(key, None)
        if value == key_value_replace[1]:
            artifact_config[key] = key_value_replace[2]
    return artifact_config

def test_evaluate(config, saved_model=None, experiment_log=None, rank_test=False):  # input pair and output the acc (have ground truth)
    resaved_experiment_log = False
    if experiment_log == None:
        experiment_log_root = config.result_dir + "exp_record.json"
        print(experiment_log_root)
        assert os.path.exists(experiment_log_root)
        experiment_log = load_dic(experiment_log_root)
        resaved_experiment_log = True
        print(f">>> test record: load experiment dic from {experiment_log_root}")

    # config.configure(batch_size=64)
    # print(config.batch_size)
    if config.return_domain_result:
        test_pred_acc, test_avg_loss, domain_acc = test_evaluate_func(config, saved_model,
                                                                      sample_saving_dir=config.result_dir,
                                                                      return_domain_acc=True
                                                                      )
        experiment_log["domain_acc"] = domain_acc
    else:
        test_pred_acc, test_avg_loss = test_evaluate_func(config, saved_model,
                                                               sample_saving_dir=config.result_dir,
                                                               return_domain_acc=False,
                                                          )
    experiment_log["test_acc"] = test_pred_acc
    experiment_log["test_loss"] = test_avg_loss

    # if rank_test:
    #     ndcg_avg, pairwise_acc_avg = test_rank_evaluate_func(config, saved_model)
    #     experiment_log["ndcg_avg"] = ndcg_avg
    #     experiment_log["pairwise_acc_avg"] = pairwise_acc_avg

    if resaved_experiment_log and not(config.no_result_saving):  # log load_from_dir
        print(f">>> Resaved it the log to {experiment_log_root}!")  # 想一下存储的方式
        with open(experiment_log_root, "w", encoding="utf-8") as fw:
            json.dump(experiment_log, fw, ensure_ascii=False)
    else:
        print(">>> due to the setting 'no_result_saving', we do not save exp_record.json here")
    return experiment_log

def test_evaluate_func(config, model=None, sample_saving_dir=None, return_domain_acc=False):
    model = load_model(config, model, pair_mode=True)
    test_loader = Dataloader.get_data_loader(config, action="test")
    if config.return_domain_result:
        assert isinstance(test_loader.dataset, MixedSampleDataset)
    else:
        assert isinstance(test_loader.dataset, SampleDataset)
    print("test dataset length:", len(test_loader))
    output = evaluation(test_loader, model, sample_saving_dir=sample_saving_dir, log_interval=1,
                        return_domain_acc=return_domain_acc)
    print(output)
    return output

# def test_rank_evaluate_func(config, model=None):
#     model = load_model(config, model, pair_mode=False)
#     rank_result_root = f"data/labeled_rank_data_collected/processed_data/rank_result_{config.domain}_{config.indicator_name}.jsonl"
#     collection_root = f"data/labeled_rank_data_collected/processed_data/rank_data_{config.domain}_{config.indicator_name}.tsv"
#     inference_rank_loader = Dataloader.get_data_loader(config, action="inference", collection_inference_root=collection_root)
#     rank_score_list = inference(inference_rank_loader, model=model, config=config)
#     rank_result_list = Loader.load_samples(rank_result_root)
#     # check _rank_list
#     sample_num, rank_num = np.array(rank_result_list).shape
#     assert len(rank_score_list) == sample_num * rank_num
#     rank_score_list = list(np.array(rank_score_list).reshape(sample_num, rank_num))
#     rank_pred_list = []
#     for rank_score in rank_score_list:
#         rank_pred = list(np.argsort(np.argsort(-np.array(rank_score))) + 1)
#         rank_pred_list.append(rank_pred)
#     ndcg_list = []
#     pairwise_acc_list = []
#     for rank_truth, rank_pred in list(zip(rank_result_list, rank_pred_list)):
#         ndcg = ndcg_func(rank_truth, rank_pred)
#         pairwise_acc = pairwise_acc_func(rank_truth, rank_pred)
#         ndcg_list.append(ndcg)
#         pairwise_acc_list.append(pairwise_acc)
#     assert len(ndcg_list) == sample_num
#     ndcg_avg = np.mean(ndcg_list)
#     pairwise_acc_avg = np.mean(pairwise_acc_list)
#     print("ndcg", ndcg_avg)
#     print("pairwise_acc", pairwise_acc_avg)
#     return ndcg_avg, pairwise_acc_avg

def ndcg_func(rank_truth, rank_pred):
    assert len(rank_truth) ==  len(rank_pred)
    rank_num = len(rank_truth)
    truth_rel = np.arange(rank_num-1,-1,-1)
    weight = np.log2(np.arange(1, rank_num+1) + 1)
    rel_truth_of_regular_seq = rank_num - 1 - np.argsort(np.array(rank_truth))  # ABCD: 2301
    rel_pred_of_regular_seq = rank_num - 1 - np.argsort(np.array(rank_pred))
    pred_rank_truth_rel = sorted(list(zip(list(rel_truth_of_regular_seq), list(rel_pred_of_regular_seq))),key=lambda x:x[1], reverse=True)
    pred_rank_truth_rel = [i[0] for i in pred_rank_truth_rel]
    ndcg = np.sum(pred_rank_truth_rel/weight) / np.sum(truth_rel/weight)
    return ndcg


# def pairwise_acc_func(rank_truth, rank_pred):
#     assert len(rank_truth) == len(rank_pred)
#     pair_idx_list = generate_pair_idx(length_of_list=len(rank_truth))
#     pairs_truth = ["-".join([str(rank_truth[pair_idx[0]]), str(rank_truth[pair_idx[1]])]) for pair_idx in pair_idx_list]
#     pairs_pred = ["-".join([str(rank_pred[pair_idx[0]]), str(rank_pred[pair_idx[1]])]) for pair_idx in pair_idx_list]
#     pairs_true = set(pairs_pred) & set(pairs_truth)
#     pairwise_acc = len(pairs_true) / len(pairs_truth)
#     return pairwise_acc

def inference_evaluate_func(config, model=None, show_sample=True):
    inference_loader = Dataloader.get_data_loader(config, action="inference")
    print("inference dataset length:", len(inference_loader))
    model = load_model(config, model=model, pair_mode=False)  # inference 由于数据格式不同，因此模型也可能产生区别
    score_list = inference(inference_loader, model=model, config=config)
    idx_value_dic = dict(zip(inference_loader.dataset.idxes, score_list))
    idx_value_result_root = config.result_dir + f"idx_value_result_{config.domain}.json"  # _{config.indicator_name}
    idx_value_result = {"collection_root": config.collection_inference_root, "idx_value_dic":idx_value_dic}
    with open(idx_value_result_root, "w", encoding="utf-8") as fw:
        json.dump(idx_value_result, fw, ensure_ascii=False)
    if show_sample:
        df = pd.DataFrame({"idx":inference_loader.dataset.idxes, "text":inference_loader.dataset.collection,
                           "value":score_list})
        df.to_excel(config.result_dir + f"sample_{config.domain}.xlsx", index=False)  # _{config.indicator_name}
    print(f">>>>>> saved idx_value_dic to {idx_value_result_root}")

if __name__ == "__main__":
    # print(pairwise_acc_func([1,2,3,4],[1, 4,3,2]))
    print(ndcg_func([1, 2, 3, 4], [1, 4,3,2]))