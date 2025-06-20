import json
from .data_loader.data_loader import Dataloader
from .evaluation import load_model, inference
from .utils.training import flatten_list_of_list




def generate_score(config, text_list, idx_list):

    inference_loader = Dataloader.get_data_loader(config, action="inference", collection=text_list, idxes=idx_list)
    print("inference dataset length:", len(inference_loader))
    model = load_model(config, pair_mode=False)  # inference 由于数据格式不同，因此模型也可能产生区别
    score_list = inference(inference_loader, model=model, config=config)
    assert len(score_list) == len(text_list)
    return score_list


if __name__ == "__main__":
    pass
