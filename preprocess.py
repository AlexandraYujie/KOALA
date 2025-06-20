from koala.data_preprocess import prepare_unrank_data
from koala.config.config import Config

# preprocess data
# prepare data/raw/raw_data.jsonl

config = Config()
prepare_unrank_data(config)