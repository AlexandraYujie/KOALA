#%%
import os.path
from koala.config.config import Config, DistillConfig
from koala.utils.training import set_seed, read_yml, amend_config_by_dic, check_config, amend_config_by_args
from koala.training.training import train
from koala.training.distill_training import get_name, distill_main
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distill", action="store_true", default=False)
    parser.add_argument("--no_test", action="store_true", default=False)
    parser.add_argument("--no_linear_add", action="store_true", default=False)
    parser.add_argument("--sigmoid_score", action="store_true", default=False)
    parser.add_argument("--inference", action="store_true", default=False)
    parser.add_argument("--load_config", action="store_true", default=False)
    parser.add_argument("--domain_mix", action="store_true", default=False)
    parser.add_argument("--few_sample_training", action="store_true", default=False)
    parser.add_argument("--no_query", action="store_true", default=False)
    parser.add_argument("--random_query", action="store_true", default=False)
    parser.add_argument("--no_model_saving", action="store_true", default=False)
    parser.add_argument("--only_encoder", action="store_true", default=False)
    parser.add_argument("--crossentropy_loss", action="store_true", default=False)
    parser.add_argument("--similarity", type=str, default="cos", help="Adaptive to sbert (method)")
    parser.add_argument("--domain", type=str)  # 传递domain参数
    parser.add_argument("--method", type=str, choices=["ColBERT", "BERT", "sbert", "T5"])  # 传递method单数
    parser.add_argument("--base_model_name", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--query", type=str)  # default = "0" # 0.tsv
    args = parser.parse_args()
    if args.load_config:
        args.config_root = f"config/predefined_config/{args.method}_config.yml"  #{args.cfg_name}_config
        assert os.path.exists(args.config_root), f"The loaded_config para has set, however, {args.config_root} not found."
    else:
        print(">>> Using the default para in setting.py and those passed in args")
    return args

def set_config(config, args):
    if args.load_config:
        loaded_config = read_yml(args.config_root)
        print("***** from predefined yml config *****")
        amend_config_by_dic(config, loaded_config)

    print("***** from args *****")
    amend_config_by_args(config, args)

def basic_setting(config, args):
    set_config(config, args)
    check_config(config)
    set_seed(config.seed)


if __name__ == "__main__":
    args = parse_args()
    if args.distill:
        config = DistillConfig()
        config.configure(
            # Domains used for distillation training.
            # For each domain, the corresponding file idx_value_result_<domain>.json will be loaded.
            domain_seq=["edu", "liv", "eco", "tech", "cul"],
            # Domains used for optional testing.
            # For each test domain, the following files are required:
            # - data/processed/test/<domain>/collection.tsv
            # - data/processed/test/<domain>/grouped_data.ranked.jsonl
            # If no test data is available, use the --no_test flag in the command line.
            mixed_domains=["edu", "eco", "liv", "tech", "cul", "med", "emp", "econ", "agri", "gov"]
        )
        basic_setting(config, args)
        distill_main(config)
    else:
        config = Config()
        basic_setting(config, args)
        train(config)

