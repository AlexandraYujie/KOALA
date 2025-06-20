from koala.evaluation import *
from koala.config.config import DistillConfig, Config
import argparse
from koala.utils.training import amend_config_by_args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument("--inference", action="store_true", default=False)
    parser.add_argument("--model_dir", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if "distill" in args.model_dir:
        config = DistillConfig()
    else:
        config = Config()

    if args.evaluate:
        configure_config(config, args.model_dir)  # config setting
        amend_config_by_args(config, args, filter_keys=["model_dir","evaluate","inference"])
        config.configure(batch_size=64)
        test_evaluate(config)
    elif args.inference:
        configure_config(config, args.model_dir)
        # amend_config_by_dic(config, args.__dict__)
        inference_evaluate_func(config)