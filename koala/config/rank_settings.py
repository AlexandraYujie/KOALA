
from koala.config.core_config import DefaultVal

class RankSetting:
    language: str = DefaultVal("cn")  # set "cn" or "en"
    style_name: str = DefaultVal("强制性")
    style_definition: str = DefaultVal("语气越强硬、措施或要求越详细具体、涉及到硬性的数字标准，则文本的“强制性”越强。")
    data_num: int = DefaultVal(500)  # limit data_num per domain, set to None when no limit
    trg_domains: set = DefaultVal({"tech", "edu"})
    train_ratio: float = DefaultVal(0.8)
    api_key: str = DefaultVal("")
    base_url: str = DefaultVal("")
    llm_model: str = DefaultVal("qwen-plus")
    max_workers: int = DefaultVal(8)
    print_resp: bool = DefaultVal(False)