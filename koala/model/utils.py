import torch
import os
import torch.nn as nn
import torch.nn.init as init


def weights_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        init.zeros_(m.bias)
def torch_load_dnn(path):
    if path.startswith("http:") or path.startswith("https:"):
        dnn = torch.hub.load_state_dict_from_url(path, map_location='cpu')
    else:
        dnn = torch.load(path, map_location='cpu')
    return dnn

def flatten(L):
    # return [x for y in L for x in y]

    result = []
    for _list in L:
        result += _list

    return result

def print_trainable_parameters(model):
    print("***"*20)
    trainable_params = 0
    all_param = 0
    for layer_name, param in model.named_parameters():
        # print(layer_name, param.requires_grad)
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print("trainable params: {} || all params: {} || trainable%: {}".format(trainable_params, all_param,
                                                                            100 * trainable_params / all_param))
    print("***" * 20)




def manage_checkpoints(model_dir, model, batch_idx, consumed_all_triples=False):
    # arguments = dict(args)

    # TODO: Call provenance() on the values that support it??

    try:
        save = model.save
    except:
        save = model.module.save

    path_save = False

    if consumed_all_triples or (batch_idx % 2000 == 0):
        # name = os.path.join(path, "colbert.dnn")
        # save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)
        path_save = model_dir

    if path_save:
        print(f"#> Saving a checkpoint to {path_save} ..")

        # checkpoint = {}
        # checkpoint['batch'] = batch_idx
        # checkpoint['epoch'] = 0
        # checkpoint['model_state_dict'] = model.state_dict()
        # checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        # checkpoint['arguments'] = arguments

        save(path_save)

    return path_save

def set_bert_grad(model, value):
    try:
        for p in model.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(model.module, value)

def get_max_len_of_queries(tok, queries, query_maxlen):
    # Ensure that query_maxlen <= length <= 500 tokens
    def max_len(length):
        return min(500, max(query_maxlen, length))
    un_truncated_ids = tok(queries, add_special_tokens=False)['input_ids']
    # Get the longest length in the batch
    max_length_in_batch = max(len(x) for x in un_truncated_ids)
    # Set the max length
    max_length = max_len(max_length_in_batch)  # 2 maybe is the cls and sep
    return max_length