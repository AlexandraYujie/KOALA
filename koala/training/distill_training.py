import warnings
from koala.data_loader.data_loader import Dataloader
from koala.evaluation import get_model
import torch
import json
import pandas as pd
from koala.data_loader.utils import Loader
from koala.model.utils import print_trainable_parameters
from transformers import AdamW, get_linear_schedule_with_warmup
from koala.model.utils import set_bert_grad
import torch.nn as nn
from koala.evaluation import test_evaluate
import copy
import numpy as np
import random
from koala.utils.training import load_dic

name_component = {"content": ["method", "base_model_name", "similarity"],
                  "bool":["only_encoder","no_linear_add","sigmoid_score","no_query","random_query"],
                  }

def get_idx_value_roots(name):
    name_idx_value_roots_dic_root = "experiment/exp_summary/name_idx_value_roots_dic.json"
    name_idx_value_roots_dic = load_dic(name_idx_value_roots_dic_root)
    idx_value_roots = name_idx_value_roots_dic.get(name, None)
    if not idx_value_roots:
        raise RuntimeError(f"no such name: {name}!")
    return idx_value_roots


# name_component = ["method","base_model_name","only_encoder","no_linear_add","sigmoid_score"]

def get_name(config):
    name_list = []
    config_dic = config.__dict__
    for component in name_component["content"]:
        assert component in config_dic
        value = config_dic[component]
        if component == "similarity" and value == "cos":
            continue
        name_list.append(value)
    for component in name_component["bool"]:
        assert component in config_dic
        if config_dic[component]:
            name_list.append(component)
    name = ".".join(name_list)
    print(name)
    return name

def get_distill_data(config):
    domain_seq = config.domain_seq
    name = get_name(config)
    idx_value_result_roots = get_idx_value_roots(name)

    idx_value_result_roots_new = []
    for domain in domain_seq:
        found = False
        for root in idx_value_result_roots:
            if domain in root:
                idx_value_result_roots_new.append(root)
                idx_value_result_roots.remove(root)
                found = True
                break
        if not found:
            warnings.warn(f"{domain} not found in idx_value_result_roots...")
    if len(idx_value_result_roots):
        warnings.warn(f"not using: {idx_value_result_roots}\n\n")
    print(idx_value_result_roots_new)
    return idx_value_result_roots_new

def load_distill_data(idx_value_result_root):
    text_value_list = []
    with open(idx_value_result_root, "r", encoding="utf-8") as fp:
        idx_value_result = json.load(fp)
    assert "collection_root" in idx_value_result and "idx_value_dic" in idx_value_result
    collection_root = idx_value_result["collection_root"]
    idx_value_dic = idx_value_result["idx_value_dic"]

    text_list, idx_list = Loader.load_inference_collection(collection_root)
    idx_list = [str(idx) for idx in idx_list]
    idx_text_dic = dict(zip(idx_list, text_list))

    distill_data_len = len(idx_text_dic)
    public_idxes = set(list(idx_text_dic.keys())) & set(list(idx_value_dic.keys()))

    assert len(public_idxes) == distill_data_len

    for idx in idx_list:
        text_value_list.append([idx_text_dic[idx], idx_value_dic[idx]])

    return text_value_list

def prepare_distill_data(text_value_list, split_proportion, shuffle=True):  # list of list, every list is one domain
    text_value_list_train = []
    text_value_list_valid = []
    for text_value_list_domain in text_value_list:
        length_domain = len(text_value_list_domain)
        split_idx = int(length_domain * split_proportion)
        train_data, valid_data= text_value_list_domain[:split_idx], text_value_list_domain[split_idx:]
        if shuffle:
            random.shuffle(train_data)
        text_value_list_train.extend(train_data)
        text_value_list_valid.extend(valid_data)
    return text_value_list_train, text_value_list_valid

def split_text_value(text_value_list):
    texts = [i[0] for i in text_value_list]
    trg_values = [i[1] for i in text_value_list]
    return {"texts":texts, "trg_values":trg_values}

def distill_main(config):
    idx_value_result_roots = get_distill_data(config)
    # config.configure(source_models=idx_value_result_roots)
    text_value_list = []
    for idx_value_result_root in idx_value_result_roots:
        text_value_list_domain = load_distill_data(idx_value_result_root)
        text_value_list.append(text_value_list_domain)

    text_value_list_train, text_value_list_valid = prepare_distill_data(text_value_list, config.proportion, config.shuffle)
    print("train data length:", len(text_value_list_train))
    print("valid data length:", len(text_value_list_valid))
    split_text_value(text_value_list_valid)
    distill_train_loader = Dataloader.get_data_loader(config, action="training", shuffle=False, **split_text_value(text_value_list_train))
    # in training action, default shuffle == True. Here, we can alter the shuffle based on our requirement
    distill_valid_loader = Dataloader.get_data_loader(config, action="validation", **split_text_value(text_value_list_valid))
    print("total batch step length:", len(distill_train_loader))

    distill_train(config, distill_train_loader, distill_valid_loader)

def distill_evaluation(data_loader, model, config, criterion):
    model.eval()
    model = model.to(config.device)
    loss_list = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            batch, trg_values = batch_data
            trg_values = torch.tensor(list(trg_values)).to(torch.device(config.device))
            scores = model(*batch)
            loss = criterion(scores, trg_values)
            loss_list.append(loss.item())
            if batch_idx % config.log_interval == 0:
                print(batch_idx, loss)
    avg_loss = sum(loss_list)/len(loss_list)
    return avg_loss

def distill_train(config, distill_train_loader, distill_valid_loader):
    experiment_log = {"epoch":[], "train_loss":[], "valid_loss":[]}
    Model = get_model(config.method)
    model = Model(config, pair_mode=False)
    model = model.to(config.device)
    saved_model = None
    criterion = nn.MSELoss()
    print_trainable_parameters(model)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, eps=1e-8)
    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup,
                                                    num_training_steps=len(distill_train_loader) * config.epoch)
    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(model, False)

    best_loss = float('inf')
    if not config.no_model_saving:
        model_dir = config.model_dir
    result_dir = config.result_dir

    for epoch in range(config.epoch):
        train_loss = 0
        print(f"*** epoch num: {epoch + 1} ***")
        model.train()
        for batch_idx, batch_data in enumerate(distill_train_loader):
            # if batch_idx >= 10:
            #     break
            optimizer.zero_grad()
            if (warmup_bert is not None) and warmup_bert <= batch_idx:
                set_bert_grad(model, True)
                warmup_bert = None
            batch, trg_values = batch_data
            trg_values = torch.tensor(list(trg_values)).to(torch.device(config.device))
            scores = model(*batch)  # forward data
            # print(scores.dtype)
            # print(trg_values.dtype)
            # trg_values = trg_values.to(torch.float16)
            loss = criterion(scores, trg_values)
            loss.backward()
            train_loss += loss
            optimizer.step()
            if scheduler:
                scheduler.step()
            if batch_idx % config.log_interval == 0:
                print(batch_idx, loss.item())

        avg_loss_of_train = train_loss / (batch_idx + 1)
        print(f">>>>>>>> epoch:{epoch+1} | training loss: {avg_loss_of_train}")

        # evaluate: valid
        if len(distill_valid_loader):
            print("\n", "===" * 10, "validation!!!!" + "===" * 10, )
            avg_loss_of_valid = distill_evaluation(distill_valid_loader, model, config, criterion)
            print(f">>>>>>>> epoch:{epoch + 1} | validation loss: {avg_loss_of_valid}")
        else:
            avg_loss_of_valid = np.nan

        if avg_loss_of_valid <= best_loss or avg_loss_of_valid != avg_loss_of_valid:
            best_loss = avg_loss_of_valid
            best_epoch = epoch + 1
            if not config.no_model_saving:
                torch.save(model.state_dict(), model_dir + "model.pt")  # file_root = dir + name
                config.save_for_checkpoint(checkpoint_path=model_dir)
                print(f">>>>>>>> epoch {epoch + 1}: saved model and config to {model_dir}")
            else:
                saved_model = copy.deepcopy(model).to(torch.device("cpu"))

        record_exp(experiment_log, epoch, avg_loss_of_train.item(), avg_loss_of_valid)


        print("\n\n\n")
    print(f"$$$$$ best model in {best_epoch} epoch | loss: {best_loss} $$$$$")

    # evaluate: test
    if not config.no_test:
        experiment_log = test_evaluate(config, saved_model=saved_model, experiment_log=experiment_log)
    save_exp_record(result_dir, experiment_log, config)
    print(f">>>>>>>> saved exp_result to {result_dir}")

def record_exp(experiment_log, epoch, train_loss, valid_loss):
    experiment_log["epoch"].append(epoch)
    experiment_log["train_loss"].append(train_loss)
    experiment_log["train_loss"].append(valid_loss)
def save_exp_record(result_dir, experiment_log, config):
    with open(result_dir + "exp_record.json", "w", encoding="utf-8") as fw:
        json.dump(experiment_log, fw, ensure_ascii=False)
    config.save_for_checkpoint(checkpoint_path=result_dir)

if __name__ == "__main__":
    get_distill_data()