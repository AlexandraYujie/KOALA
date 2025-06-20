from .settings import TrainSetting, TokenizerSettings, ResourceSetting, DistillSetting, DataSetting
from .anal_setting import AnalSetting
from .base_config import BaseConfig
from .rank_settings import RankSetting
from dataclasses import dataclass

@dataclass
class Config(TrainSetting, BaseConfig, TokenizerSettings, ResourceSetting):
    pass

@dataclass
class DistillConfig(DistillSetting, Config):
    pass

@dataclass
class AnalConfig(AnalSetting, BaseConfig):
    pass

@dataclass
class RankConfig(RankSetting, DataSetting, BaseConfig):
    pass