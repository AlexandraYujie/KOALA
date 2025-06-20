import copy
import json
from ..evaluation import print_progress, evaluation, get_model, test_evaluate, inference_evaluate_func
from koala.data_loader.data_loader import Dataloader
from koala.model.utils import manage_checkpoints, print_trainable_parameters
from transformers import AdamW, get_linear_schedule_with_warmup
from koala.model.utils import set_bert_grad
import torch
from torch import nn


def train(config):
    experiment_log = {"epoch":[], "train_acc":[], "valid_acc":[]}
    training_loader = Dataloader.get_data_loader(config, action="training")
    validation_loader = Dataloader.get_data_loader(config, action="validation")

    Model = get_model(config.method)
    model = Model(config)
    model = model.to(config.device)
    # print_trainable_parameters(model)
    saved_model = None
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, eps=1e-8)
    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup,
                                                    num_training_steps=len(training_loader) * config.epoch)
    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(model, False)

    labels = torch.zeros(config.batch_size, dtype=torch.long, device=config.device)
    best_evaluate_pred_acc = 0
    # train!
    if not config.no_model_saving:
        model_dir = config.model_dir

    for epoch in range(config.epoch):
        print(f"*** epoch num: {epoch + 1} ***")
        model.train()
        pred_acc_list = []
        for batch_idx, batch_data in enumerate(training_loader):
            optimizer.zero_grad()
            if (warmup_bert is not None) and warmup_bert <= batch_idx:
                set_bert_grad(model, True)
                warmup_bert = None
            if training_loader.append_value:
                batch, batch_domain = batch_data
            else:
                batch = batch_data
            scores = model(*batch)  # forward data
            scores = scores.view(-1, 2)  # [4,2]
            if config.crossentropy_loss:
                loss = nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])
            else:
                loss = torch.log(1 + torch.exp(scores[:, 1] - scores[:, 0])).sum()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            pred_acc_list_batch, positive_avg, negative_avg = print_progress(scores)
            pred_acc_list.extend(pred_acc_list_batch)
            if batch_idx % config.log_interval == 0:
                print("#>>>   ", positive_avg, negative_avg, '\t\t|\t\t', positive_avg - negative_avg)
                print(">> pred acc for this batch (training):", sum(pred_acc_list_batch) / len(pred_acc_list_batch))
                print(batch_idx, loss.item())

        pred_acc = sum(pred_acc_list) / len(pred_acc_list)  # for every epoch
        print(">> pred acc for this epoch (training):", pred_acc)

        print("\n","==="*10, "validation!!!!" + "==="*10, )
        valid_pred_acc, evaluate_avg_loss = evaluation(validation_loader, model)
        if valid_pred_acc >= best_evaluate_pred_acc:
            best_evaluate_pred_acc = valid_pred_acc
            if not config.no_model_saving:
                torch.save(model.state_dict(), model_dir + "model.pt")  # file_root = dir + name
                config.save_for_checkpoint(checkpoint_path=model_dir)
                # manage_checkpoints(model_dir, model, batch_idx + 1,
                #                                consumed_all_triples=True)  # return model_dir
                print(f">>>>>>>> epoch {epoch + 1}: saved model and config to {model_dir}")
            else:
                saved_model = copy.deepcopy(model).to(torch.device("cpu"))
            best_epoch = epoch + 1
        print("\n\n\n")
        record_exp(experiment_log, epoch, pred_acc, valid_pred_acc)

    print(f"$$$$$ best model in {best_epoch} epoch | acc: {best_evaluate_pred_acc} $$$$$")

    # evaluating for test data
    config.configure(batch_size=32)
    print(config.batch_size)

    if not config.no_test:
        experiment_log = test_evaluate(config, saved_model, experiment_log)

    if not config.no_result_saving:
        save_exp_record(config.result_dir, experiment_log, config)
        print(f">>>>>>>> saved exp_result to {config.result_dir}")
        print("\n\n\n\n\n")
    else:
        print(f">>> due to the setting 'no_result_saving', we do not save exp_record.json in {config.result_dir}")

    if config.inference:
        inference_evaluate_func(config, model=saved_model)



def record_exp(experiment_log, epoch, train_acc, valid_acc):
    experiment_log["epoch"].append(epoch)
    experiment_log["train_acc"].append(train_acc)
    experiment_log["valid_acc"].append(valid_acc)


def save_exp_record(result_path, experiment_log, config):
    with open(result_path + "exp_record.json", "w", encoding="utf-8") as fw:
        json.dump(experiment_log, fw, ensure_ascii=False)
    config.save_for_checkpoint(checkpoint_path=result_path)  # export to artifact.metadata





